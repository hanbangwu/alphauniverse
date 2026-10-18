<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import { IMAGE_MODES, SPECTRUM_MODES } from '$lib/data/modes'
  import { getApp } from '$lib/state/app.svelte'
  import MaskControls from './mask-controls.svelte'
  import MatchRow from './match-row.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const images = $derived(IMAGE_MODES.some((mode) => similarity.anywhere(mode)))
  const spectra = $derived(SPECTRUM_MODES.some((mode) => similarity.anywhere(mode)))
  const predicted = $derived(similarity.predictedAt(0))
</script>

<span class="text-sm font-medium">Whole dataset</span>
<p class="text-xs text-muted-foreground">
  Image matches are compared {images ? 'anywhere in the image' : 'at the same place in the image'};
  spectrum matches {spectra ? 'at any wavelength' : 'at the same observed wavelength'}.
  {#if similarity.wholeResult}
    Each match's score is the mean over the chosen modes of its standardised score in each; the bars
    show each mode's share. Maps compare each slot at the same place.
  {/if}
</p>
{#if similarity.wholeResult && predicted.length}
  <p class="text-xs text-muted-foreground">
    #{similarity.galaxy} has no {predicted.join(', ')}: searched with AION's prediction
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
