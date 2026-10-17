<script lang="ts">
  import type { SpectrumSurvey } from '$lib/client'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { spectrumQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { SURVEYS } from '$lib/labels'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    survey: SpectrumSurvey
  }

  let { galaxy, survey }: Props = $props()

  const spectrum = createQuery(() => spectrumQuery(galaxy, survey))
</script>

<PanelFrame busy={spectrum.isFetching} class="aspect-2/1 bg-card">
  {#if spectrum.data}
    <SpectrumChart
      spectrum={spectrum.data}
      label={`${SURVEYS[survey].label} spectrum of galaxy ${galaxy}`}
      class="absolute inset-0"
    />
  {/if}
</PanelFrame>

{#if spectrum.isError}
  <p class="text-xs leading-relaxed text-destructive">
    {errorMessage(spectrum.error)}
  </p>
{/if}
