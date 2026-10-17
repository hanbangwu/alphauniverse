<script lang="ts" module>
  import GalaxyTooltip from './galaxy-tooltip.svelte'
  import type { DataPoint } from 'embedding-atlas'
  import { mount, unmount } from 'svelte'

  interface TooltipProps {
    tooltip: DataPoint
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
    }

    destroy(): void {
      void unmount(this.#component)
    }
  }
</script>

<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import { colouringQuery, morphologiesQuery, similarityQuery } from '$lib/data/queries'
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
  const full = createQuery(() => app.fullPoints)
  const search = createQuery(() => similarityQuery(view.searched.value))

  const loaded = $derived((view.full ? full.data : mean.data) ?? null)
  const colouring = createQuery(() =>
    colouringQuery(mosaic, loaded, view.property, view.source.value)
  )
  const coloured = $derived(view.property ? (colouring.data ?? null) : null)
  const table = $derived(view.property ? (coloured?.table ?? null) : loaded)
  const faded = $derived(view.source.value === 'measured' && view.property !== 'g_r')

  let tooltip: DataPoint | number | null = $derived(view.galaxy.value)

  const results = $derived(
    search.data && !search.isPlaceholderData && !view.full ? Array.from(search.data.galaxies) : null
  )
  const circled = $derived(view.galaxy.value === null ? null : (results ?? [view.galaxy.value]))
  const morphologies = createQuery(() => morphologiesQuery(mosaic, mean.data ?? null, circled))

  let highlighted = $derived(
    circled?.filter(
      (galaxy) =>
        !filters.active ||
        filters.morphologies.value.some((index) => index === morphologies.data?.get(galaxy))
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
        additionalFields={view.property
          ? {
              property: { sql: `'${COLOURINGS[view.property].label}'` },
              value: 'value',
              width: 'width',
              predicted: 'predicted'
            }
          : null}
        identifier="galaxy"
        filter={mosaic.filter}
        {tooltip}
        selection={highlighted}
        {width}
        {height}
        config={{ colorScheme: mode.current, autoLabelEnabled: false }}
        theme={{ statusBar: false }}
        customTooltip={Tooltip}
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
    {#if view.property && coloured}
      <ColourLegend property={view.property} {coloured} {faded} />
    {/if}
  {:else}
    <div class="absolute inset-0 grid place-content-center">
      <Spinner class="size-8" />
    </div>
  {/if}
</div>
