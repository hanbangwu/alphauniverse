<script lang="ts">
  import { SELECTED } from '$lib/color'
  import { Spinner } from '$lib/components/ui/spinner'
  import { positionsQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { getApp } from '$lib/state/app.svelte'
  import GalaxyTooltip from './galaxy-tooltip.svelte'
  import { createQuery } from '@tanstack/svelte-query'
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
    galaxy: number
    patch: number | null
  }

  const TILES = 'https://www.legacysurvey.org/viewer/ls-dr10-south/1/{z}/{x}/{y}.jpg'
  const CREDIT = '<a href="https://www.legacysurvey.org/viewer">Legacy Surveys</a>'
  const NATIVE_ZOOM = 14
  const PATCH_ZOOM = 16
  const MAX_ZOOM = 17
  const REACH = 6
  const DOT_RADIUS = 5
  const PULSE_SECONDS = 1
  const FOCUSED = ['any', ...['hovered', 'selected'].map(focus)]
  const PHASE = ['%', ['/', ['time'], PULSE_SECONDS], 1]
  const EASED = ['-', 1, ['^', ['-', 1, PHASE], 3]]
  const SKY_STYLE = [
    {
      style: [
        {
          'circle-radius': ['*', DOT_RADIUS, ['+', 1, EASED]],
          'circle-fill-color': 'white',
          'circle-opacity': ['*', 0.6, ['-', 1, PHASE]]
        },
        { 'circle-radius': DOT_RADIUS, 'circle-fill-color': 'white' }
      ]
    },
    {
      filter: FOCUSED,
      style: {
        'circle-radius': DOT_RADIUS + 1.5,
        'circle-fill-color': 'rgba(0, 0, 0, 0)',
        'circle-stroke-color': SELECTED,
        'circle-stroke-width': 2.5
      }
    }
  ]
  const MARKS: Record<string, Style> = {
    hovered: new Style({ fill: new Fill({ color: 'rgba(255, 255, 255, 0.4)' }) }),
    chosen: new Style({
      fill: new Fill({ color: 'rgba(239, 68, 68, 0.35)' }),
      stroke: new Stroke({ color: SELECTED, width: 1 })
    })
  }
  const GRIDS = {
    other: gridStyle('rgba(255, 255, 255, 0.25)', 0),
    open: gridStyle('rgba(255, 255, 255, 0.6)', 1)
  }
  const [HALF_STAMP] = fromLonLat([(96 * 0.262) / 2 / 3600, 0])

  const { view, meta } = getApp()
  const { grid } = meta

  const positions = createQuery(() => positionsQuery)

  const places = $derived(
    positions.data &&
      Array.from(positions.data.ra, (ra, galaxy) =>
        fromLonLat([180 - ra, positions.data.dec[galaxy]])
      )
  )

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

  function gridStyle(color: string, zIndex: number): Style {
    return new Style({
      geometry: (feature) => gridLines((feature.getGeometry() as Point).getCoordinates()),
      stroke: new Stroke({ color, width: 1 }),
      zIndex
    })
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
    style: SKY_STYLE,
    variables: { hovered: -1, selected: -1 },
    maxZoom: PATCH_ZOOM - 0.001
  })
  const squares = new VectorLayer({
    source: galaxies,
    style: (feature) => (feature.getId() === view.galaxy.value ? GRIDS.open : GRIDS.other),
    minZoom: PATCH_ZOOM - 0.001
  })

  function patchTarget(feature: Feature<Point> | null, coordinate: Coordinate): Target | null {
    if (!feature) return null
    const point = feature.getGeometry()!
    return containsCoordinate(buffer(point.getExtent(), HALF_STAMP), coordinate)
      ? { galaxy: feature.getId() as number, patch: patchAt(point.getCoordinates(), coordinate) }
      : null
  }

  function pick(map: Map, pixel: Pixel, coordinate: Coordinate): Target | null {
    const closest = galaxies.getClosestFeatureToCoordinate(coordinate)
    if (!closest) return null
    if ((map.getView().getZoom() ?? 0) >= PATCH_ZOOM) {
      const open = view.galaxy.value
      return (
        (open === null ? null : patchTarget(galaxies.getFeatureById(open), coordinate)) ??
        patchTarget(closest, coordinate)
      )
    }
    const [x, y] = map.getPixelFromCoordinate(closest.getGeometry()!.getCoordinates())
    return Math.hypot(pixel[0] - x, pixel[1] - y) <= REACH
      ? { galaxy: closest.getId() as number, patch: null }
      : null
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
        squares,
        new VectorLayer({
          source: marks,
          style: (feature) => MARKS[feature.get('kind')],
          minZoom: PATCH_ZOOM - 0.001
        })
      ],
      overlays: [overlay],
      controls: defaults({ zoom: false, rotate: false, attributionOptions: { collapsible: false } })
    })
    rings.on('postrender', () => {
      if ((created.getView().getZoom() ?? 0) < PATCH_ZOOM) created.render()
    })
    created.on('pointermove', (event) => {
      if (event.dragging) return
      hover(pick(created, event.pixel, event.coordinate))
      overlay.setPosition(hovered?.patch === null ? event.coordinate : undefined)
    })
    created.on('click', (event) => {
      const picked = pick(created, event.pixel, event.coordinate)
      if (!picked) return
      view.select(picked.galaxy)
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
    if (!places) return
    galaxies.clear(true)
    galaxies.addFeatures(
      places.map((place, galaxy) => {
        const feature = new Feature({ geometry: new Point(place), galaxy })
        feature.setId(galaxy)
        return feature
      })
    )
  })

  $effect(() => {
    rings.updateStyleVariables({
      hovered: hovered?.galaxy ?? -1,
      selected: view.galaxy.value ?? -1
    })
    squares.changed()
  })

  $effect(() => {
    marks.clear()
    if (!places) return
    if (hovered?.patch != null) {
      marks.addFeature(
        new Feature({
          geometry: fromExtent(patchArea(places[hovered.galaxy], hovered.patch)),
          kind: 'hovered'
        })
      )
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
    <GalaxyTooltip tooltip={{ x: 0, y: 0, identifier: hovered.galaxy }} />
  {/if}
</div>

{#if positions.error}
  <p
    role="alert"
    class="pointer-events-none absolute inset-0 grid place-content-center text-sm text-destructive"
  >
    {errorMessage(positions.error)}
  </p>
{:else if !places}
  <div class="pointer-events-none absolute inset-0 grid place-content-center">
    <Spinner class="size-8" />
  </div>
{/if}
