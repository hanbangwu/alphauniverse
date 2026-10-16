<script lang="ts">
  import CollapsibleGroup from '$lib/components/common/collapsible-group.svelte'
  import SimilarityDialog from '$lib/components/similarity/similarity-dialog.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Spinner } from '$lib/components/ui/spinner'
  import { IMAGE_SURVEYS, SPECTRUM_SURVEYS } from '$lib/data/modes'
  import { coverageQuery, morphologyQuery } from '$lib/data/queries'
  import { MORPHOLOGIES, SURVEYS } from '$lib/labels'
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
  const coverage = createQuery(() => coverageQuery(view.galaxy.value))
</script>

<div class="flex h-full flex-col bg-card">
  <h2 class="shrink-0 p-4 pb-2 text-xs font-medium tracking-wider text-muted-foreground uppercase">
    Details
  </h2>

  <div class="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 pb-4">
    {#if view.galaxy.value === null}
      <p class="text-sm leading-relaxed text-muted-foreground">No galaxy selected.</p>
    {:else}
      {@const galaxy = view.galaxy.value}
      <div class="flex items-baseline justify-between text-sm">
        <span class="font-medium">#{galaxy}</span>
        {#if morphology.isPending}
          <Spinner class="size-3 self-center text-muted-foreground" />
        {:else}
          <span class="text-xs text-muted-foreground">
            {(morphology.data == null ? null : MORPHOLOGIES[morphology.data]) ?? '-'}
          </span>
        {/if}
      </div>

      <Button variant="secondary" size="sm" onclick={() => (view.explorer.value = true)}>
        Search
      </Button>

      <CollapsibleGroup title="Images" bind:open={view.open.current.images}>
        {#each IMAGE_SURVEYS as survey (survey)}
          <div class="flex flex-col gap-1">
            <span class="text-xs text-muted-foreground">{SURVEYS[survey].label}</span>
            {#if coverage.data?.[survey]}
              <DetailImage {galaxy} {survey} />
            {:else if coverage.data}
              <p class="text-xs text-muted-foreground">No image</p>
            {/if}
          </div>
        {/each}
      </CollapsibleGroup>

      <CollapsibleGroup title="Spectra" bind:open={view.open.current.spectra}>
        {#each SPECTRUM_SURVEYS as survey (survey)}
          <div class="flex flex-col gap-1">
            <span class="text-xs text-muted-foreground">{SURVEYS[survey].label}</span>
            {#if coverage.data?.[survey]}
              <DetailSpectrum {galaxy} {survey} />
            {:else if coverage.data}
              <p class="text-xs text-muted-foreground">No spectrum</p>
            {/if}
          </div>
        {/each}
      </CollapsibleGroup>

      <CoverageTable {galaxy} />

      {#key galaxy}
        <SimilarityDialog {galaxy} />
      {/key}
    {/if}
  </div>
</div>
