<script lang="ts">
  import { BINS, SELECTED, propertyColors } from '$lib/color'
  import { Spinner } from '$lib/components/ui/spinner'
  import {
    type Shown,
    colouredQuery,
    colouringQuery,
    pointsQuery,
    positionsQuery
  } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { PROPERTIES } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import GalaxyTooltip from './galaxy-tooltip.svelte'
  import { createQuery } from '@tanstack/svelte-query'
  import { scaleLinear } from 'd3-scale'
  import { mode } from 'mode-watcher'
  import Feature from 'ol/Feature'
  import Map from 'ol/Map'
  import Overlay from 'ol/Overlay'
  import View from 'ol/View'
  import { defaults } from 'ol/control/defaults'
  import type { Coordinate } from 'ol/coordinate'
  import type { Extent } from 'ol/extent'
  import { boundingExtent, buffer, containsCoordinate, getCenter } from 'ol/extent'
  import { MultiLineString, Point } from 'ol/geom'
  import { fromExtent } from 'ol/geom/Polygon'
  import TileLayer from 'ol/layer/Tile'
  import VectorLayer from 'ol/layer/Vector'
  import WebGLVectorLayer from 'ol/layer/WebGLVector'
  import 'ol/ol.css'
  import type { Pixel } from 'ol/pixel'
  import { fromLonLat } from 'ol/proj'
  import VectorSource from 'ol/source/Vector'
  import XYZ from 'ol/source/XYZ'
  import { Fill, Stroke, Style } from 'ol/style'
  import { onMount } from 'svelte'

  interface Target {
    shown: Shown
    patch: number | null
  }

  const TILES = 'https://www.legacysurvey.org/viewer/ls-dr10-south/1/{z}/{x}/{y}.jpg'
  const CREDIT = '<a href="https://www.legacysurvey.org/viewer">Legacy Surveys</a>'
  const NATIVE_ZOOM = 14
  const PATCH_ZOOM = 16
  const MAX_ZOOM = 17
  const RING_ZOOMS = [10, 14, 16]
  const RING_RADII = [1.5, 10, 30]
  const MINIMUM_REACH = 6
  const FOCUSED = ['any', ...['hovered', 'selected'].map(focus)]
  const RING = {
    'circle-radius': [
      'interpolate',
      ['linear'],
      ['zoom'],
      ...RING_ZOOMS.flatMap((zoom, index) => [zoom, RING_RADII[index]])
    ],
    'circle-fill-color': 'rgba(0, 0, 0, 0)',
    'circle-stroke-color': ['get', 'colour'],
    'circle-stroke-width': ['case', FOCUSED, 2.5, 1],
    'circle-opacity': ['case', FOCUSED, 1, 0.4]
  }
  const MARKS: Record<string, Style> = {
    grid: new Style({ stroke: new Stroke({ color: 'rgba(255, 255, 255, 0.35)', width: 1 }) }),
    hovered: new Style({ fill: new Fill({ color: 'rgba(255, 255, 255, 0.4)' }) }),
    chosen: new Style({
      fill: new Fill({ color: 'rgba(239, 68, 68, 0.35)' }),
      stroke: new Stroke({ color: SELECTED, width: 1 })
    })
  }
  const SQUARE = new Style({
    geometry: (feature) =>
      fromExtent(buffer((feature.getGeometry() as Point).getExtent(), HALF_STAMP)),
    stroke: new Stroke({ color: 'rgba(255, 255, 255, 0.5)', width: 1 })
  })
  const [HALF_STAMP] = fromLonLat([(96 * 0.262) / 2 / 3600, 0])

  const { view, mosaic, filters, meta } = getApp()
  const { grid } = meta
  const ringRadius = scaleLinear(RING_ZOOMS, RING_RADII).clamp(true)

  const positions = createQuery(() => positionsQuery)
  const shown = createQuery(() => pointsQuery(mosaic, view.pointSet.value))
  const colouring = createQuery(() =>
    colouringQuery(mosaic, shown.data ?? null, view.property.value, view.everywhere)
  )
  const coloured = createQuery(() => colouredQuery(mosaic, colouring.data?.name ?? null))

  const places = $derived(
    positions.data &&
      Array.from(positions.data.ra, (ra, galaxy) =>
        fromLonLat([180 - ra, positions.data.dec[galaxy]])
      )
  )

  function inRange({ value }: Shown): boolean {
    const range = filters.range.value
    return !range || (value !== null && value >= range[0] && value <= range[1])
  }

  function focus(name: string) {
    return ['==', ['get', 'galaxy'], ['var', name]]
  }

  function patchArea(place: Coordinate, patch: number): Extent {
    const size = (2 * HALF_STAMP) / grid
    const left = place[0] - HALF_STAMP + (patch % grid) * size
    const top = place[1] + HALF_STAMP - Math.floor(patch / grid) * size
    return [left, top - size, left + size, top]
  }

  function patchAt(place: Coordinate, [x, y]: Coordinate): number {
    const size = (2 * HALF_STAMP) / grid
    const column = Math.floor((x - place[0] + HALF_STAMP) / size)
    const row = Math.floor((place[1] + HALF_STAMP - y) / size)
    return row * grid + column
  }

  function gridLines([x, y]: Coordinate): MultiLineString {
    const steps = Array.from(
      { length: grid + 1 },
      (_, step) => -HALF_STAMP + (2 * HALF_STAMP * step) / grid
    )
    return new MultiLineString(
      steps.flatMap((step) => [
        [
          [x + step, y - HALF_STAMP],
          [x + step, y + HALF_STAMP]
        ],
        [
          [x - HALF_STAMP, y + step],
          [x + HALF_STAMP, y + step]
        ]
      ])
    )
  }

  let element: HTMLDivElement
  let tip: HTMLDivElement
  let map = $state<Map | null>(null)
  let width = $state(0)
  let framed = false
  let hovered = $state<Target | null>(null)

  const galaxies = new VectorSource<Feature<Point>>()
  const marks = new VectorSource()
  const rings = new WebGLVectorLayer({
    source: galaxies,
    style: RING,
    variables: { hovered: -1, selected: -1 }
  })

  function pick(map: Map, pixel: Pixel, coordinate: Coordinate): Target | null {
    const closest = galaxies.getClosestFeatureToCoordinate(coordinate)
    if (!closest) return null
    const shown = closest.get('shown') as Shown
    const place = closest.getGeometry()!.getCoordinates()
    const zoom = map.getView().getZoom() ?? 0
    if (zoom >= PATCH_ZOOM) {
      const area = buffer(closest.getGeometry()!.getExtent(), HALF_STAMP)
      return containsCoordinate(area, coordinate)
        ? { shown, patch: patchAt(place, coordinate) }
        : null
    }
    const [x, y] = map.getPixelFromCoordinate(place)
    const reach = Math.max(ringRadius(zoom), MINIMUM_REACH)
    return Math.hypot(pixel[0] - x, pixel[1] - y) <= reach ? { shown, patch: null } : null
  }

  function hover(next: Target | null): void {
    hovered = next
    element.style.cursor = next ? 'pointer' : ''
  }

  onMount(() => {
    const overlay = new Overlay({ element: tip, offset: [16, 16], stopEvent: false })
    const created = new Map({
      target: element,
      layers: [
        new TileLayer({
          source: new XYZ({ url: TILES, maxZoom: NATIVE_ZOOM, attributions: CREDIT })
        }),
        rings,
        new VectorLayer({ source: galaxies, style: SQUARE, minZoom: PATCH_ZOOM - 0.001 }),
        new VectorLayer({
          source: marks,
          style: (feature) => MARKS[feature.get('kind')],
          minZoom: PATCH_ZOOM - 0.001
        })
      ],
      overlays: [overlay],
      controls: defaults({ zoom: false, rotate: false, attributionOptions: { collapsible: false } })
    })
    created.on('pointermove', (event) => {
      if (event.dragging) return
      hover(pick(created, event.pixel, event.coordinate))
      overlay.setPosition(hovered?.patch === null ? event.coordinate : undefined)
    })
    created.on('click', (event) => {
      const picked = pick(created, event.pixel, event.coordinate)
      if (!picked) return
      view.select(picked.shown.galaxy)
      if (picked.patch !== null) view.tokens.ls_image.toggle(picked.patch)
    })
    const leave = () => {
      hover(null)
      overlay.setPosition(undefined)
    }
    element.addEventListener('pointerleave', leave)
    map = created
    return () => {
      element.removeEventListener('pointerleave', leave)
      created.setTarget(undefined)
    }
  })

  $effect(() => {
    if (!map || !places || !width || framed) return
    framed = true
    const area = buffer(boundingExtent(places), 2 * HALF_STAMP)
    const sky = new View({
      center: getCenter(area),
      maxZoom: MAX_ZOOM,
      extent: area,
      showFullExtent: true,
      enableRotation: false
    })
    map.updateSize()
    map.setView(sky)
    sky.fit(area)
  })

  $effect(() => {
    if (!places || !mode.current) return
    const shades = propertyColors(mode.current)
    const rows =
      coloured.data ??
      places.map((_, galaxy): Shown => ({
        galaxy,
        colour: 2 * BINS,
        value: null,
        width: null,
        predicted: null
      }))
    galaxies.clear(true)
    galaxies.addFeatures(
      rows.filter(inRange).map(
        (row) =>
          new Feature({
            geometry: new Point(places[row.galaxy]),
            colour: shades[row.colour],
            galaxy: row.galaxy,
            shown: row
          })
      )
    )
  })

  $effect(() => {
    rings.updateStyleVariables({
      hovered: hovered?.shown.galaxy ?? -1,
      selected: view.galaxy.value ?? -1
    })
  })

  $effect(() => {
    marks.clear()
    if (!places) return
    if (hovered?.patch != null) {
      const place = places[hovered.shown.galaxy]
      marks.addFeatures([
        new Feature({ geometry: gridLines(place), kind: 'grid' }),
        new Feature({ geometry: fromExtent(patchArea(place, hovered.patch)), kind: 'hovered' })
      ])
    }
    const open = view.galaxy.value
    if (open === null) return
    marks.addFeatures(
      view.tokens.ls_image.value.map(
        (patch) =>
          new Feature({ geometry: fromExtent(patchArea(places[open], patch)), kind: 'chosen' })
      )
    )
  })

  const failed = $derived(positions.error ?? colouring.error ?? coloured.error)
</script>

<div
  bind:this={element}
  bind:clientWidth={width}
  role="application"
  aria-label="Sky map of the galaxies in the dataset"
  class="size-full bg-black"
></div>

<div bind:this={tip} class="pointer-events-none">
  {#if hovered?.patch === null}
    <GalaxyTooltip
      tooltip={{
        x: 0,
        y: 0,
        identifier: hovered.shown.galaxy,
        fields: {
          value: hovered.shown.value,
          width: hovered.shown.width,
          predicted: hovered.shown.predicted
        }
      }}
      label={PROPERTIES[view.property.value].label}
    />
  {/if}
</div>

{#if failed}
  <p
    role="alert"
    class="pointer-events-none absolute inset-0 grid place-content-center text-sm text-destructive"
  >
    {errorMessage(failed)}
  </p>
{:else if !places}
  <div class="pointer-events-none absolute inset-0 grid place-content-center">
    <Spinner class="size-8" />
  </div>
{/if}
