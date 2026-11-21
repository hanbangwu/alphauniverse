<script lang="ts">
  import type { SpectrumSurvey } from '$lib/client'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { spectrumQuery, spectrumAionPredictionQuery } from '$lib/data/queries'
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

  const predicted = $derived(view.dataSource.value === 'prediction')
  const spectrum = createQuery(() => spectrumQuery(matched && !predicted ? galaxy : null, survey))
  const prediction = createQuery(() =>
    spectrumAionPredictionQuery(predicted ? galaxy : null, survey)
  )
  const drawn = $derived(predicted ? prediction.data : spectrum.data)
  const label = $derived(SURVEYS[survey])
</script>

<PanelFrame busy={spectrum.isFetching || prediction.isFetching} class="aspect-2/1 bg-card">
  {#if drawn}
    <SpectrumChart
      spectrum={drawn}
      normalized={predicted && !matched}
      label={predicted
        ? `AION's predicted ${label} spectrum of galaxy ${galaxy}${matched ? '' : ', normalized'}`
        : `${label} spectrum of galaxy ${galaxy}`}
      class="absolute inset-0"
    />
  {/if}
</PanelFrame>

{#if spectrum.isError}
  <p class="text-xs leading-relaxed text-destructive">
    {spectrum.error.message}
  </p>
{/if}
{#if prediction.isError}
  <p class="text-xs leading-relaxed text-destructive">
    {prediction.error.message}
  </p>
{/if}
