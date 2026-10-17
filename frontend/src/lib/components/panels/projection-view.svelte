<script lang="ts" module>
  import GalaxyTooltip from './galaxy-tooltip.svelte'
  import type { DataPoint } from 'embedding-atlas'
  import { mount, unmount } from 'svelte'

  interface TooltipProps {
    tooltip: DataPoint
    label: string | null
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
  import {
    colouringQuery,
    morphologiesQuery,
    similarityQuery,
    valuesQuery
  } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { COLOURINGS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import ColourLegend from './colour-legend.svelte'
  import { createQuery } from '@tanstack/svelte-query'
  import { EmbeddingViewMosaic } from 'embedding-atlas/svelte'
  import { mode } from 'mode-watcher'

  const app = getApp()
  const { view, mosaic, filters } = app

  let width = $state(0)
  let height = $state(0)

  const mean = createQuery(() => app.meanPoints)
  const shown = createQuery(() => app.shownPoints)
  const search = createQuery(() => similarityQuery(view.searched.value))

  const loaded = $derived(shown.data ?? null)
  const colouring = createQuery(() =>
    colouringQuery(mosaic, loaded, view.property, view.everywhere)
  )
  const coloured = $derived(view.property ? (colouring.data ?? null) : null)
  const table = $derived(view.property ? (coloured?.table ?? null) : loaded)

  let tooltip: DataPoint | number | null = $derived(view.galaxy.value)

  const results = $derived(
    search.data && !search.isPlaceholderData && !view.full ? Array.from(search.data.galaxies) : null
  )
  const circled = $derived(view.galaxy.value === null ? null : (results ?? [view.galaxy.value]))
  const morphologies = createQuery(() => morphologiesQuery(mosaic, mean.data ?? null, circled))
  const values = createQuery(() =>
    valuesQuery(mosaic, filters.range.value && coloured ? coloured.name : null, circled)
  )

  function inRange(galaxy: number): boolean {
    const range = filters.range.value
    if (!range || !coloured) return true
    const value = values.data?.get(galaxy)
    return value != null && value >= range[0] && value <= range[1]
  }

  let highlighted = $derived(
    circled?.filter(
      (galaxy) =>
        (!filters.active ||
          filters.morphologies.value.some((index) => index === morphologies.data?.get(galaxy))) &&
        inRange(galaxy)
    )
  )
</script>

<div class="relative size-full overflow-hidden" bind:clientWidth={width} bind:clientHeight={height}>
  {#if table && app.swatches && app.shades && mode.current && width > 0 && height > 0}
    {#key table}
      <EmbeddingViewMosaic
        coordinator={mosaic.coordinator}
        {table}
        x="x"
        y="y"
        category={coloured ? 'colour' : 'category'}
        categoryColors={coloured ? app.shades : app.swatches}
        additionalFields={coloured
          ? { value: 'value', width: 'width', predicted: 'predicted' }
          : null}
        identifier="galaxy"
        filter={coloured ? mosaic.colouredFilter : mosaic.filter}
        {tooltip}
        selection={highlighted}
        {width}
        {height}
        config={{ colorScheme: mode.current, autoLabelEnabled: false }}
        theme={{ statusBar: false }}
        customTooltip={{
          class: Tooltip,
          props: { label: coloured ? COLOURINGS[coloured.property].label : null }
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
    {#if coloured}
      {#key coloured.name}
        <ColourLegend {coloured} scheme={mode.current} />
      {/key}
    {/if}
  {:else if view.property && colouring.isError}
    <p role="alert" class="absolute inset-0 grid place-content-center text-sm text-destructive">
      {errorMessage(colouring.error)}
    </p>
  {:else}
    <div class="absolute inset-0 grid place-content-center">
      <Spinner class="size-8" />
    </div>
  {/if}
</div>
