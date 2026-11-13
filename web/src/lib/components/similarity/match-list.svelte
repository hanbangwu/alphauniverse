<script lang="ts">
  import type { GetModeSearchDownloadData, GetSearchDownloadData } from '$lib/api'
  import { client } from '$lib/client'
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Spinner } from '$lib/components/ui/spinner'
  import { isModeQuery } from '$lib/data/similarity'
  import { must } from '$lib/invariant'
  import { getApp } from '$lib/state/app.svelte'
  import MaskControls from './mask-controls.svelte'
  import MatchRow from './match-row.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const predicted = $derived(similarity.predictedAt(0))
  const download = $derived.by(() => {
    const { request } = must(similarity.result, 'the search result')
    return isModeQuery(request)
      ? client.buildUrl<GetModeSearchDownloadData>({
          url: '/search/modes/download',
          query: request
        })
      : client.buildUrl<GetSearchDownloadData>({ url: '/search/download', query: request })
  })
</script>

<div class="flex items-center justify-between gap-2">
  <span class="text-sm font-medium">Whole dataset</span>
  {#if similarity.scales && !similarity.error}
    <Button
      variant="secondary"
      size="sm"
      disabled={similarity.stale}
      href={download}
      download="alphauniverse-search.ecsv">Export as ECSV</Button
    >
  {/if}
</div>
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
