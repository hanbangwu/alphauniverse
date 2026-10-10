<script lang="ts">
  import { tokenAlpha, tokenColors } from '$lib/color'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { coverageQuery, spectrumQuery, spectrumTokensQuery } from '$lib/data/queries'
  import { hasSpectrum } from '$lib/data/spectra'
  import { errorMessage } from '$lib/errors'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    map: Float32Array | null
    selected?: number[]
    ontoggle?: (index: number) => void
    absent?: string
  }

  let { galaxy, map, selected, ontoggle, absent = 'No spectrum' }: Props = $props()

  const similarity = getSimilarity()

  const coverage = createQuery(() => coverageQuery(galaxy))
  const matched = $derived(coverage.data ? hasSpectrum(coverage.data) : false)
  const spectrum = createQuery(() => spectrumQuery(matched ? galaxy : null))
  const tokens = createQuery(() => spectrumTokensQuery(matched && !map ? galaxy : null))

  const cells = $derived(map ?? tokens.data ?? null)
  const palette = $derived(
    map ? similarity.spectrumHeat : tokens.data ? tokenColors(tokens.data) : null
  )
</script>

<div class="flex flex-col gap-2">
  <span class="text-sm font-medium">Spectrum Tokens</span>

  <PanelFrame
    busy={coverage.isPending || spectrum.isFetching || tokens.isFetching}
    class="aspect-2/1 bg-card md:aspect-4/1"
  >
    {#if spectrum.data && cells && palette}
      <SpectrumChart
        spectrum={spectrum.data}
        values={cells}
        color={palette}
        opacity={map ? undefined : tokenAlpha}
        title={map ? similarity.score : similarity.caption}
        {selected}
        {ontoggle}
        label={ontoggle
          ? 'Scroll to zoom, drag to pan, click a spectrum token to select it'
          : `Spectrum of galaxy ${galaxy}`}
        class={ontoggle ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
      />
    {:else if coverage.data && !matched}
      <p class="absolute inset-0 grid place-content-center text-xs text-muted-foreground">
        {absent}
      </p>
    {/if}
  </PanelFrame>
  {#if spectrum.isError}
    <p role="alert" class="text-xs leading-relaxed text-destructive">
      {errorMessage(spectrum.error)}
    </p>
  {/if}
</div>
