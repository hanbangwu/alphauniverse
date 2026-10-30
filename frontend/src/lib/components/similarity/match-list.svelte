<script lang="ts">
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Spinner } from '$lib/components/ui/spinner'
  import { tableQuery } from '$lib/data/queries'
  import { galaxyRows, saveEcsv } from '$lib/download'
  import { errorMessage } from '$lib/errors'
  import { must } from '$lib/invariant'
  import { getApp } from '$lib/state/app.svelte'
  import MaskControls from './mask-controls.svelte'
  import MatchRow from './match-row.svelte'
  import { searchEcsv } from './search-export'
  import { getSimilarity } from './similarity.svelte'
  import { useQueryClient } from '@tanstack/svelte-query'

  const similarity = getSimilarity()
  const { view, mask: display, meta } = getApp()
  const queryClient = useQueryClient()

  const predicted = $derived(similarity.predictedAt(0))

  let downloadError = $state<string | null>(null)

  async function download(): Promise<void> {
    const result = must(similarity.result, 'the search result')
    try {
      const [identities, tableRows] = await Promise.all([
        galaxyRows(queryClient, Array.from(result.galaxies)),
        queryClient.fetchQuery(tableQuery(result.request.galaxy))
      ])
      saveEcsv(
        'alphauniverse-search.ecsv',
        searchEcsv(result, meta, identities, tableRows, location.href)
      )
      downloadError = null
    } catch (caught) {
      downloadError = errorMessage(caught)
    }
  }
</script>

<div class="flex items-center justify-between gap-2">
  <span class="text-sm font-medium">Whole dataset</span>
  {#if similarity.scales && !similarity.error}
    <Button variant="secondary" size="sm" disabled={similarity.stale} onclick={download}
      >Export as ECSV</Button
    >
  {/if}
</div>
{#if downloadError}
  <p role="alert" class="text-sm text-destructive">{downloadError}</p>
{/if}
{#if similarity.wholeResult}
  <p class="text-xs text-muted-foreground">
    Each match's score is the mean over the chosen modes of its standardised score in each; the bars
    show each mode's share. Maps compare each slot at the same place.
  </p>
{/if}
{#if similarity.wholeResult && predicted.length}
  <p class="text-xs text-muted-foreground">
    <GalaxyName galaxy={similarity.galaxy} /> has no {predicted.join(', ')}: searched with AION's
    prediction
  </p>
{/if}

{#if similarity.error}
  <p role="alert" class="text-sm text-destructive">{similarity.error}</p>
{:else if !similarity.scales}
  <div class="grid h-24 place-content-center">
    <Spinner class="size-6" />
  </div>
{:else}
  {#if display.on.value}
    <div class="flex flex-wrap gap-x-8 gap-y-3">
      {#if view.open.current.images && similarity.imageDomain}
        <MaskControls
          label="Image mask threshold"
          domain={similarity.imageDomain}
          control={display.images.matches}
        />
      {/if}
      {#if view.open.current.spectra && similarity.spectrumDomain}
        <MaskControls
          label="Spectrum mask threshold"
          domain={similarity.spectrumDomain}
          control={display.spectra.matches}
        />
      {/if}
    </div>
  {/if}
  <div class={['flex flex-col transition-opacity', similarity.stale && 'opacity-50']}>
    {#each similarity.matches as { galaxy, index } (galaxy)}
      <MatchRow {galaxy} {index} />
    {/each}
  </div>
{/if}
