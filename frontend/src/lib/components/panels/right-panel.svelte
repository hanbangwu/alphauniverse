<script lang="ts">
  import Segmented from '$lib/components/common/segmented.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Spinner } from '$lib/components/ui/spinner'
  import { morphologyQuery } from '$lib/data/queries'
  import { MORPHOLOGIES } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import CoverageTable from './coverage-table.svelte'
  import DetailImage from './detail-image.svelte'
  import DetailSpectrum from './detail-spectrum.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  const app = getApp()
  const { view, mosaic } = app

  const points = createQuery(() => app.meanPoints)
  const morphology = createQuery(() =>
    morphologyQuery(mosaic, points.data ?? null, view.galaxy.value)
  )
</script>

<div class="flex h-full flex-col bg-card">
  <h2 class="shrink-0 p-4 pb-2 text-xs font-medium tracking-wider text-muted-foreground uppercase">
    Details
  </h2>

  <div class="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 pb-4">
    {#if view.galaxy.value === null}
      <p class="text-sm leading-relaxed text-muted-foreground">No galaxy selected.</p>
    {:else}
      {@const galaxy = view.galaxy.value}
      <Segmented field={view.detail} />

      {#if view.detail.value === 'spectrum'}
        <DetailSpectrum {galaxy} />
      {:else}
        <DetailImage {galaxy} />
      {/if}

      {#await import('$lib/components/similarity/patch-similarity.svelte')}
        <Button variant="secondary" size="sm" disabled>Search</Button>
      {:then { default: PatchSimilarity }}
        <Button variant="secondary" size="sm" onclick={() => (view.explorer.value = true)}>
          Search
        </Button>
        {#key galaxy}
          <PatchSimilarity {galaxy} />
        {/key}
      {/await}

      <div class="flex items-baseline justify-between text-sm">
        <span class="font-medium">Galaxy #{galaxy}</span>
        {#if morphology.isPending}
          <Spinner class="size-3 self-center text-muted-foreground" />
        {:else}
          <span class="text-xs text-muted-foreground">
            {(morphology.data == null ? null : MORPHOLOGIES[morphology.data]) ?? '-'}
          </span>
        {/if}
      </div>

      <CoverageTable {galaxy} />
    {/if}
  </div>
</div>
