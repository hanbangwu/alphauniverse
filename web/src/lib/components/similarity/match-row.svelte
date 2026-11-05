<script lang="ts">
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import DetailImage from '$lib/components/panels/detail-image.svelte'
  import { IMAGE_SURVEYS, SPECTRUM_SURVEYS, imageMode, spectrumMode } from '$lib/data/modes'
  import { SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import ContributionBars from './contribution-bars.svelte'
  import ImageTokenPanel from './image-token-panel.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'

  interface Props {
    galaxy: number
    index: number
  }

  let { galaxy, index }: Props = $props()

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const score = $derived(similarity.scoreAt(index))
  const predicted = $derived(similarity.predictedAt(index))
</script>

<div class="flex flex-col gap-5 pt-6">
  <div class="flex flex-col gap-2">
    <button
      type="button"
      onclick={() => view.select(galaxy)}
      class="w-fit cursor-pointer rounded-sm text-left text-sm font-medium underline-offset-4 outline-none hover:underline focus-visible:ring-2 focus-visible:ring-ring/50"
    >
      <GalaxyName {galaxy} />: {similarity.total(score)}<span class="sr-only"
        >, search from this galaxy</span
      >
    </button>
    {#if view.open.current.images}
      <div class="flex max-w-2xl flex-col gap-5">
        {#each IMAGE_SURVEYS as survey (survey)}
          {#if similarity.shows(survey, index)}
            {@const mode = imageMode(survey)}
            {@const values = similarity.mapAt(mode, index)}
            <div class="grid items-end gap-5 md:grid-cols-2">
              <DetailImage {galaxy} {survey} />
              <ImageTokenPanel
                label={SURVEYS[survey].label}
                {values}
                grid={similarity.grid}
                color={similarity.scales?.[mode]?.heat ?? null}
                mask={similarity.maskOf(values, similarity.imageDomain, display.images.matches)}
                title={similarity.score}
                selected={similarity.outlinesAt(mode, index)}
              />
            </div>
          {/if}
        {/each}
      </div>
    {/if}
  </div>
  {#if similarity.wholeResult}
    <ContributionBars {index} />
  {/if}
  {#if predicted.length}
    <p class="text-xs text-muted-foreground">
      No {predicted.join(', ')}: matched on AION's prediction
    </p>
  {/if}
  {#if view.open.current.spectra}
    {#each SPECTRUM_SURVEYS as survey (survey)}
      {#if similarity.shows(survey, index)}
        {@const mode = spectrumMode(survey)}
        {@const values = similarity.mapAt(mode, index)}
        <SpectrumPanel
          {galaxy}
          {survey}
          matched={similarity.observes(survey, index)}
          map={values}
          mask={similarity.maskOf(values, similarity.spectrumDomain, display.spectra.matches)}
          heat={similarity.scales?.[mode]?.heat ?? null}
          caption={similarity.score}
          selected={similarity.outlinesAt(mode, index)}
        />
      {/if}
    {/each}
  {/if}
</div>
