<script lang="ts">
  import type { SpectrumSurvey } from '$lib/client'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { spectrumQuery, spectrumTemplateQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { SURVEYS } from '$lib/labels'
  import { getView } from '$lib/state/app.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    survey: SpectrumSurvey
    matched: boolean
  }

  let { galaxy, survey, matched }: Props = $props()

  const view = getView()

  const spectrum = createQuery(() => spectrumQuery(matched ? galaxy : null, survey))
  const prediction = createQuery(() =>
    spectrumTemplateQuery(view.predictions.value ? galaxy : null, survey)
  )
  const label = $derived(SURVEYS[survey].label)
</script>

<PanelFrame busy={spectrum.isFetching || prediction.isFetching} class="aspect-2/1 bg-card">
  {#if matched ? spectrum.data : prediction.data}
    <SpectrumChart
      spectrum={spectrum.data}
      prediction={prediction.data}
      label={matched
        ? `${label} spectrum of galaxy ${galaxy}${prediction.data ? ", with AION's prediction" : ''}`
        : `AION's predicted ${label} spectrum of galaxy ${galaxy}, normalised`}
      class="absolute inset-0"
    />
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
