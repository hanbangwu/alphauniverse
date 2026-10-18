<script lang="ts" module>
  import GalaxyTooltip from './galaxy-tooltip.svelte'
  import type { DataPoint, Point, Rectangle } from 'embedding-atlas'
  import { mount, unmount } from 'svelte'

  interface TooltipProps {
    tooltip: DataPoint
    label: string
  }

  class Tooltip {
    #props: TooltipProps
    #component: ReturnType<typeof mount>

    constructor(target: HTMLDivElement, props: TooltipProps) {
      this.#props = $state(props)
      this.#component = mount(GalaxyTooltip, { target, props: this.#props })
    }

    update(props: TooltipProps): void {
      this.#props.tooltip = props.tooltip
      this.#props.label = props.label
    }

    destroy(): void {
      void unmount(this.#component)
    }
  }
</script>

<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import { colouringQuery, regionQuery, similarityQuery, valuesQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { PROPERTIES } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import ColourLegend from './colour-legend.svelte'
  import MapSelection from './map-selection.svelte'
  import { createQuery } from '@tanstack/svelte-query'
  import { EmbeddingViewMosaic } from 'embedding-atlas/svelte'
  import { mode } from 'mode-watcher'

  const app = getApp()
  const { view, mosaic, filters } = app

  let width = $state(0)
  let height = $state(0)

  const shown = createQuery(() => app.shownPoints)
  const search = createQuery(() => similarityQuery(view.searched.value))

  const loaded = $derived(shown.data ?? null)
  const colouring = createQuery(() =>
    colouringQuery(mosaic, loaded, view.property.value, view.everywhere)
  )
  const coloured = $derived(colouring.data ?? null)

  let drawn = $state<{ points: string; value: Rectangle | Point[] } | null>(null)
  const region = $derived(drawn && drawn.points === loaded ? drawn.value : null)
  const selected = createQuery(() =>
    regionQuery(mosaic, coloured?.table ?? null, region, filters.range.value)
  )
  let settling: ReturnType<typeof setTimeout> | undefined

  let tooltip: DataPoint | number | null = $derived(view.galaxy.value)

  const results = $derived(
    search.data && !search.isPlaceholderData ? Array.from(search.data.galaxies) : null
  )
  const circled = $derived(view.galaxy.value === null ? null : (results ?? [view.galaxy.value]))
  const values = createQuery(() =>
    valuesQuery(mosaic, filters.range.value ? (coloured?.name ?? null) : null, circled)
  )

  function inRange(galaxy: number): boolean {
    const range = filters.range.value
    if (!range) return true
    const value = values.data?.get(galaxy)
    return value != null && value >= range[0] && value <= range[1]
  }

  let highlighted = $derived(circled?.filter(inRange))
</script>

<div class="relative size-full overflow-hidden" bind:clientWidth={width} bind:clientHeight={height}>
  {#if coloured && app.shades && mode.current && width > 0 && height > 0}
    {#key coloured.table}
      <EmbeddingViewMosaic
        coordinator={mosaic.coordinator}
        table={coloured.table}
        x="x"
        y="y"
        category="colour"
        categoryColors={app.shades}
        additionalFields={{ value: 'value', width: 'width', predicted: 'predicted' }}
        identifier="galaxy"
        filter={mosaic.filter}
        rangeSelectionValue={region}
        onRangeSelection={(value: Rectangle | Point[] | null) => {
          clearTimeout(settling)
          settling = setTimeout(
            () => (drawn = value && loaded ? { points: loaded, value } : null),
            200
          )
        }}
        {tooltip}
        selection={highlighted}
        {width}
        {height}
        config={{ colorScheme: mode.current, autoLabelEnabled: false }}
        theme={{ statusBar: false }}
        customTooltip={{
          class: Tooltip,
          props: { label: PROPERTIES[coloured.property].label }
        }}
        onTooltip={(point: DataPoint | null) => (tooltip = point)}
        onSelection={(picked: DataPoint[] | null) => {
          const galaxies = picked?.map((point) => Number(point.identifier)) ?? []
          const galaxy = galaxies.at(-1) ?? null
          const toggled = galaxies.length > 1 && highlighted?.some((circled) => circled === galaxy)
          if (picked && !toggled) view.select(galaxy)
          highlighted = highlighted && [...highlighted]
        }}
      />
    {/key}
    {#if region && selected.isError}
      <p role="alert" class="absolute top-3 right-3 text-xs text-destructive">
        {errorMessage(selected.error)}
      </p>
    {:else if region && selected.data}
      <MapSelection galaxies={selected.data} onclear={() => (drawn = null)} />
    {:else}
      <p class="pointer-events-none absolute top-3 right-3 text-xs text-muted-foreground">
        Shift-drag to select a box; Shift-⌘-drag (Shift-Windows key), a lasso
      </p>
    {/if}
    {#key coloured.name}
      <ColourLegend {coloured} scheme={mode.current} />
    {/key}
  {:else if shown.isError || colouring.isError}
    <p role="alert" class="absolute inset-0 grid place-content-center text-sm text-destructive">
      {errorMessage(shown.error ?? colouring.error)}
    </p>
  {:else}
    <div class="absolute inset-0 grid place-content-center">
      <Spinner class="size-8" />
    </div>
  {/if}
</div>
