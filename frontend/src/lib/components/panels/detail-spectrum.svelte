<script lang="ts">
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { coverageQuery, spectrumQuery, spectrumTemplateQuery } from '$lib/data/queries'
  import { SPECTRUM_SURVEY, hasSpectrum } from '$lib/data/spectra'
  import { errorMessage } from '$lib/errors'
  import { SURVEYS } from '$lib/labels'
  import { getView } from '$lib/state/app.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const view = getView()

  const coverage = createQuery(() => coverageQuery(galaxy))
  const matched = $derived(coverage.data ? hasSpectrum(coverage.data) : false)
  const spectrum = createQuery(() => spectrumQuery(matched ? galaxy : null))
  const prediction = createQuery(() =>
    spectrumTemplateQuery(view.predictions.value ? galaxy : null)
  )
  const observed = $derived(matched ? (spectrum.data ?? null) : null)
  const predicted = $derived(prediction.data ?? null)
  const survey = SURVEYS[SPECTRUM_SURVEY].label
</script>

<PanelFrame
  busy={coverage.isPending || spectrum.isFetching || prediction.isFetching}
  class="bg-card"
>
  {#if matched ? observed : predicted}
    <SpectrumChart
      spectrum={observed}
      prediction={predicted}
      label={observed
        ? `${survey} spectrum of galaxy ${galaxy}${predicted ? ", with AION's prediction" : ''}`
        : `AION's predicted ${survey} spectrum of galaxy ${galaxy}, normalised`}
      class="absolute inset-0"
    />
  {:else if coverage.data && !matched && !view.predictions.value}
    <p class="absolute inset-0 grid place-content-center text-xs text-muted-foreground">
      No spectrum
    </p>
  {/if}
</PanelFrame>

{#if spectrum.isError}
  <p class="text-xs leading-relaxed text-destructive">
    {errorMessage(spectrum.error)}
  </p>
{/if}
{#if prediction.isError}
  <p class="text-xs leading-relaxed text-destructive">
    {errorMessage(prediction.error)}
  </p>
{/if}
