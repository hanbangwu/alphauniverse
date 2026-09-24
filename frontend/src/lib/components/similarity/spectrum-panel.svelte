<script lang="ts">
  import { tokenColors } from '$lib/color'
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { coverageQuery, spectrumQuery, spectrumTokensQuery } from '$lib/data/queries'
  import { spectrumSurvey } from '$lib/data/spectra'
  import { SURVEYS } from '$lib/labels'
  import { getMeta } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    map: Float32Array | null
    selected?: number[]
    onselect?: (indices: number[]) => void
  }

  let { galaxy, map, selected, onselect }: Props = $props()

  const meta = getMeta()
  const similarity = getSimilarity()

  const coverage = createQuery(() => coverageQuery(meta, galaxy))
  const survey = $derived(coverage.data ? spectrumSurvey(coverage.data) : null)
  const spectrum = createQuery(() => spectrumQuery(meta, galaxy, survey))
  const tokens = createQuery(() => spectrumTokensQuery(meta, galaxy, map ? null : survey))

  const cells = $derived(map ?? tokens.data ?? null)
  const palette = $derived(
    map ? similarity.spectrumHeat : tokens.data ? tokenColors(tokens.data) : null
  )
  const caption = $derived(map ? similarity.score : (value: number) => `token ${value}`)
</script>

<div class="flex flex-1 flex-col gap-2">
  <div class="flex items-baseline justify-between">
    <span class="text-sm font-medium">Spectrum</span>
    {#if survey}
      <span class="text-xs text-muted-foreground">{SURVEYS[survey]?.label ?? survey}</span>
    {/if}
  </div>

  <PatchFrame busy={coverage.isPending || spectrum.isFetching || tokens.isFetching} class="bg-card">
    {#if spectrum.data && cells && palette}
      <SpectrumChart
        spectrum={spectrum.data}
        values={cells}
        color={palette}
        title={caption}
        {selected}
        {onselect}
        label={onselect
          ? 'Scroll to zoom, drag to pan, click a span to select its token'
          : `Spectrum of galaxy ${galaxy}`}
        class={onselect ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
      />
    {:else if coverage.data && !survey}
      <p class="absolute inset-0 grid place-content-center text-xs text-muted-foreground">
        No spectrum
      </p>
    {/if}
  </PatchFrame>
</div>
