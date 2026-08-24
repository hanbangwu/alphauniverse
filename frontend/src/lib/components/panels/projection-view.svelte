<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import { getApp } from '$lib/state/app.svelte'
  import { createQuery } from '@tanstack/svelte-query'
  import type { DataPoint } from 'embedding-atlas'
  import { EmbeddingViewMosaic } from 'embedding-atlas/svelte'
  import { mode } from 'mode-watcher'

  const app = getApp()
  const { view, mosaic } = app

  let width = $state(0)
  let height = $state(0)

  const mean = createQuery(() => app.meanPoints)
  const full = createQuery(() => app.fullPoints)

  const loaded = $derived((view.full ? full.data : mean.data) ?? null)

  const highlighted = $derived(view.galaxy.value === null ? null : [view.galaxy.value])
</script>

<div class="relative size-full overflow-hidden" bind:clientWidth={width} bind:clientHeight={height}>
  {#if loaded && app.swatches && mode.current && width > 0 && height > 0}
    {#key loaded}
      <EmbeddingViewMosaic
        coordinator={mosaic.coordinator}
        table={loaded}
        x="x"
        y="y"
        category="category"
        categoryColors={app.swatches}
        identifier="galaxy"
        filter={mosaic.filter}
        selection={highlighted}
        {width}
        {height}
        config={{ colorScheme: mode.current }}
        theme={{ statusBar: false }}
        onSelection={(picked: DataPoint[] | null) => {
          const galaxy = picked?.at(-1)?.identifier
          view.select(galaxy === undefined ? null : Number(galaxy))
        }}
      />
    {/key}
  {:else}
    <div class="absolute inset-0 grid place-content-center">
      <Spinner class="size-8" />
    </div>
  {/if}
</div>
