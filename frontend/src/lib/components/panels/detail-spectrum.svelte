<script lang="ts">
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { coverageQuery, spectrumQuery } from '$lib/data/queries'
  import { spectrumSurvey } from '$lib/data/spectra'
  import { SURVEYS } from '$lib/labels'
  import { getMeta } from '$lib/state/app.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const meta = getMeta()

  const coverage = createQuery(() => coverageQuery(meta, galaxy))
  const survey = $derived(coverage.data ? spectrumSurvey(coverage.data) : null)
  const spectrum = createQuery(() => spectrumQuery(meta, galaxy, survey))
</script>

<PatchFrame busy={coverage.isPending || spectrum.isFetching} class="bg-card">
  {#if survey && spectrum.data}
    <SpectrumChart
      spectrum={spectrum.data}
      label={`${SURVEYS[survey]?.label ?? survey} spectrum of galaxy ${galaxy}`}
      class="absolute inset-0"
    />
    <span class="absolute top-1 right-2 text-[10px] text-muted-foreground">
      {SURVEYS[survey]?.label ?? survey}
    </span>
  {:else if coverage.data && !survey}
    <p class="absolute inset-0 grid place-content-center text-xs text-muted-foreground">
      No spectrum
    </p>
  {/if}
</PatchFrame>

{#if spectrum.isError}
  <p class="text-xs leading-relaxed text-destructive">
    {'detail' in spectrum.error ? spectrum.error.detail : spectrum.error.message}
  </p>
{/if}
