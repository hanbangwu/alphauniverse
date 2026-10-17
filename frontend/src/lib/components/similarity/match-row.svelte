<script lang="ts">
  import { IMAGE_SURVEYS, SPECTRUM_SURVEYS, imageMode, spectrumMode } from '$lib/data/modes'
  import { SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import ContributionBars from './contribution-bars.svelte'
  import GalaxyTile from './galaxy-tile.svelte'
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
  {#if view.open.current.images}
    <div class="grid max-w-5xl gap-5 md:grid-cols-3">
      <GalaxyTile {galaxy} {score} onselect={() => view.select(galaxy)} />
      {#each IMAGE_SURVEYS as survey (survey)}
        {#if similarity.shows(survey, index)}
          {@const mode = imageMode(survey)}
          {@const values = similarity.mapAt(mode, index)}
          <div class="flex flex-col gap-5">
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
  {:else}
    <span class="text-sm font-medium">#{galaxy}: {similarity.total(score)}</span>
  {/if}
  {#if similarity.wholeModes}
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
