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
  import { mode } from 'mode-watcher'
  import Feature from 'ol/Feature'
  import Map from 'ol/Map'
  import Overlay from 'ol/Overlay'
  import View from 'ol/View'
  import { defaults } from 'ol/control/defaults'
  import type { Coordinate } from 'ol/coordinate'
  import { boundingExtent, buffer, containsCoordinate, getCenter } from 'ol/extent'
  import { Point } from 'ol/geom'
  import { fromExtent } from 'ol/geom/Polygon'
  import TileLayer from 'ol/layer/Tile'
  import VectorLayer from 'ol/layer/Vector'
  import WebGLVectorLayer from 'ol/layer/WebGLVector'
  import 'ol/ol.css'
  import { fromLonLat } from 'ol/proj'
  import VectorSource from 'ol/source/Vector'
  import XYZ from 'ol/source/XYZ'
  import { onMount } from 'svelte'

  const TILES = 'https://www.legacysurvey.org/viewer/ls-dr10-south/1/{z}/{x}/{y}.jpg'
  const CREDIT = '<a href="https://www.legacysurvey.org/viewer">Legacy Surveys</a>'
  const NATIVE_ZOOM = 14
  const MAX_ZOOM = 16
  const RING = {
    'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 1.5, 14, 10, 16, 30],
    'circle-fill-color': 'rgba(0, 0, 0, 0)',
    'circle-stroke-color': ['get', 'colour'],
    'circle-stroke-width': 1.5
  }
  const [HALF_STAMP] = fromLonLat([(96 * 0.262) / 2 / 3600, 0])

  const { view, mosaic, filters } = getApp()

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

  function stamp(place: Coordinate): Feature {
    return new Feature(fromExtent(buffer(boundingExtent([place]), HALF_STAMP)))
  }

  let element: HTMLDivElement
  let tip: HTMLDivElement
  let map = $state<Map | null>(null)
  let width = $state(0)
  let framed = false
  let hovered = $state<Shown | null>(null)

  const galaxies = new VectorSource<Feature<Point>>()
  const hoveredStamp = new VectorSource()
  const selectedStamp = new VectorSource()
  const rings = new WebGLVectorLayer({ source: galaxies, style: RING })

  function pick(coordinate: Coordinate): Shown | null {
    const closest = galaxies.getClosestFeatureToCoordinate(coordinate)
    if (!closest) return null
    const area = buffer(closest.getGeometry()!.getExtent(), HALF_STAMP)
    return containsCoordinate(area, coordinate) ? (closest.get('shown') as Shown) : null
  }

  function hover(next: Shown | null): void {
    hovered = next
    hoveredStamp.clear()
    if (next && places) hoveredStamp.addFeature(stamp(places[next.galaxy]))
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
        new VectorLayer({
          source: hoveredStamp,
          style: { 'stroke-color': 'white', 'stroke-width': 1 }
        }),
        new VectorLayer({
          source: selectedStamp,
          style: { 'stroke-color': SELECTED, 'stroke-width': 2 }
        })
      ],
      overlays: [overlay],
      controls: defaults({ zoom: false, rotate: false, attributionOptions: { collapsible: false } })
    })
    created.on('pointermove', (event) => {
      if (event.dragging) return
      hover(pick(event.coordinate))
      overlay.setPosition(hovered ? event.coordinate : undefined)
    })
    created.on('click', (event) => {
      const picked = pick(event.coordinate)
      if (picked) view.select(picked.galaxy)
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
            shown: row
          })
      )
    )
  })

  $effect(() => {
    selectedStamp.clear()
    if (places && view.galaxy.value !== null) {
      selectedStamp.addFeature(stamp(places[view.galaxy.value]))
    }
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
  {#if hovered}
    <GalaxyTooltip
      tooltip={{
        x: 0,
        y: 0,
        identifier: hovered.galaxy,
        fields: { value: hovered.value, width: hovered.width, predicted: hovered.predicted }
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
