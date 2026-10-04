<script lang="ts">
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import { coverageQuery, spectrumQuery } from '$lib/data/queries'
  import { SPECTRUM_SURVEY, hasSpectrum } from '$lib/data/spectra'
  import { SURVEYS } from '$lib/labels'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const coverage = createQuery(() => coverageQuery(galaxy))
  const matched = $derived(coverage.data ? hasSpectrum(coverage.data) : false)
  const spectrum = createQuery(() => spectrumQuery(matched ? galaxy : null))
  const chart = import('$lib/components/spectrum/spectrum-chart.svelte')
</script>

<PatchFrame busy={coverage.isPending || spectrum.isFetching} class="bg-card">
  {#if matched && spectrum.data}
    {#await chart then { default: SpectrumChart }}
      <SpectrumChart
        spectrum={spectrum.data}
        label={`${SURVEYS[SPECTRUM_SURVEY].label} spectrum of galaxy ${galaxy}`}
        class="absolute inset-0"
      />
    {/await}
  {:else if coverage.data && !matched}
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
