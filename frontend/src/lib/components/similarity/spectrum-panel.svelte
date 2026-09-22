<script lang="ts">
  import { tokenColors } from '$lib/color'
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { coverageQuery, spectrumQuery, spectrumTokensQuery } from '$lib/data/queries'
  import { spectrumSurvey } from '$lib/data/spectra'
  import { SURVEYS } from '$lib/labels'
  import { getMeta, getView } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  const meta = getMeta()
  const view = getView()
  const { galaxy } = getSimilarity()

  const coverage = createQuery(() => coverageQuery(meta, galaxy))
  const survey = $derived(coverage.data ? spectrumSurvey(coverage.data) : null)
  const spectrum = createQuery(() => spectrumQuery(meta, galaxy, survey))
  const tokens = createQuery(() => spectrumTokensQuery(meta, galaxy, survey))

  const palette = $derived(tokens.data ? tokenColors(tokens.data) : null)
</script>

<div class="flex flex-1 flex-col gap-2">
  <div class="flex items-baseline justify-between">
    <span class="text-sm font-medium">Spectrum</span>
    {#if survey}
      <span class="text-xs text-muted-foreground">{SURVEYS[survey]?.label ?? survey}</span>
    {/if}
  </div>

  <PatchFrame busy={coverage.isPending || spectrum.isFetching || tokens.isFetching} class="bg-card">
    {#if spectrum.data && tokens.data && palette}
      <SpectrumChart
        spectrum={spectrum.data}
        tokens={tokens.data}
        color={palette}
        selected={view.spans.value}
        onselect={(indices) => (view.spans.value = indices)}
        label="Scroll to zoom, drag to pan, click a span to select its token"
        class="absolute inset-0 cursor-pointer"
      />
    {:else if coverage.data && !survey}
      <p class="absolute inset-0 grid place-content-center text-xs text-muted-foreground">
        No spectrum
      </p>
    {/if}
  </PatchFrame>
</div>
