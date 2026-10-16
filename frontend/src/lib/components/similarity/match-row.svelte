<script lang="ts">
  import { IMAGE_SURVEYS, SPECTRUM_SURVEYS, imageMode, spectrumMode } from '$lib/data/modes'
  import { OBSERVATIONS, SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import GalaxyTile from './galaxy-tile.svelte'
  import ImageTokenMask from './image-token-mask.svelte'
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
  const predicted = $derived(similarity.predictedAt(index).map((column) => OBSERVATIONS[column]))
</script>

<div class="flex flex-col gap-5 pt-6">
  {#if view.open.current.images}
    <div class="grid max-w-5xl gap-5 md:grid-cols-3">
      <GalaxyTile {galaxy} {score} />
      {#each IMAGE_SURVEYS as survey (survey)}
        {#if similarity.observes(survey, index)}
          {@const mode = imageMode(survey)}
          {@const values = similarity.mapAt(mode, index)}
          <div class="flex flex-col gap-5">
            <ImageTokenPanel
              label={SURVEYS[survey].label}
              {values}
              grid={similarity.grid}
              color={similarity.scales?.[mode]?.heat ?? null}
              title={similarity.score}
              selected={similarity.outlinesAt(mode, index)}
            />
            <ImageTokenMask {values} domain={similarity.imageDomain} control={display.matches} />
          </div>
        {/if}
      {/each}
    </div>
  {:else}
    <span class="text-sm font-medium">#{galaxy}: {similarity.score(score)}</span>
  {/if}
  {#if predicted.length}
    <p class="text-xs text-muted-foreground">
      No {predicted.join(', ')}: matched on AION's prediction
    </p>
  {/if}
  {#if view.open.current.spectra}
    {#each SPECTRUM_SURVEYS as survey (survey)}
      {#if similarity.observes(survey, index)}
        {@const mode = spectrumMode(survey)}
        <SpectrumPanel
          {galaxy}
          {survey}
          map={similarity.mapAt(mode, index)}
          heat={similarity.scales?.[mode]?.heat ?? null}
          caption={similarity.score}
          selected={similarity.outlinesAt(mode, index)}
        />
      {/if}
    {/each}
  {/if}
</div>
