<script lang="ts">
  import type { GalaxyId } from '$lib/client'
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import { IMAGE_SURVEYS, SPECTRUM_SURVEYS, imageMode, spectrumMode } from '$lib/data/modes'
  import { MODES, SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import { DECIMALS } from '$lib/state/mask.svelte'
  import ImagePatchPanel from './image-patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'

  interface Props {
    galaxy: GalaxyId
    index: number
  }

  let { galaxy, index }: Props = $props()

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const score = $derived(similarity.scoreAt(index))
  const predicted = $derived(similarity.predictedAt(index))
  const half = (value: number) =>
    similarity.contributionLimit > 0 ? (50 * Math.abs(value)) / similarity.contributionLimit : 0
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
      <div class="grid max-w-2xl gap-5 md:grid-cols-2">
        {#each IMAGE_SURVEYS as survey (survey)}
          {#if similarity.shows(survey, index)}
            {@const mode = imageMode(survey)}
            {@const values = similarity.mapAt(mode, index)}
            <section class="flex flex-col gap-2">
              <h3 class="text-xs tracking-wider text-muted-foreground uppercase">
                {SURVEYS[survey]}
              </h3>
              <ImagePatchPanel
                label={SURVEYS[survey]}
                {values}
                color={similarity.scales?.[mode]?.heat ?? null}
                mask={similarity.maskOf(values, similarity.imageDomain, display.images.matches)}
                title={similarity.score}
                {galaxy}
                {survey}
                selected={similarity.outlinesAt(mode, index)}
              />
            </section>
          {/if}
        {/each}
      </div>
    {/if}
  </div>
  {#if similarity.wholeResult}
    <dl class="grid max-w-md grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-1 text-xs">
      {#each similarity.contributionModes as mode (mode)}
        {@const value = similarity.contributionAt(mode, index)}
        <dt class="text-muted-foreground">{MODES[mode]}</dt>
        <dd class="relative h-2 rounded-sm bg-muted" aria-hidden="true">
          <div class="absolute inset-y-0 left-1/2 w-px bg-foreground/40"></div>
          <div
            class={['absolute inset-y-0', value < 0 ? 'bg-destructive' : 'bg-primary']}
            style:left="{value < 0 ? 50 - half(value) : 50}%"
            style:width="{half(value)}%"
          ></div>
        </dd>
        <dd class="text-right font-mono tabular-nums">{value.toFixed(DECIMALS)}</dd>
      {/each}
    </dl>
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
        <section class="flex flex-col gap-2">
          <h3 class="text-xs tracking-wider text-muted-foreground uppercase">{SURVEYS[survey]}</h3>
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
        </section>
      {/if}
    {/each}
  {/if}
</div>
