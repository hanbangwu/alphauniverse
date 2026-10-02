<script lang="ts">
  import { tokenAlpha, tokenColors } from '$lib/color'
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { coverageQuery, spectrumQuery, spectrumTokensQuery } from '$lib/data/queries'
  import { hasSpectrum } from '$lib/data/spectra'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    map: Float32Array | null
    selected?: number[]
    onselect?: (indices: number[]) => void
  }

  let { galaxy, map, selected, onselect }: Props = $props()

  const similarity = getSimilarity()

  const coverage = createQuery(() => coverageQuery(galaxy))
  const matched = $derived(coverage.data ? hasSpectrum(coverage.data) : false)
  const spectrum = createQuery(() => spectrumQuery(matched ? galaxy : null))
  const tokens = createQuery(() => spectrumTokensQuery(matched && !map ? galaxy : null))

  const cells = $derived(map ?? tokens.data ?? null)
  const palette = $derived(
    map ? similarity.spectrumHeat : tokens.data ? tokenColors(tokens.data) : null
  )
  const caption = $derived(map ? similarity.score : (value: number) => `token ${value}`)
</script>

<div class="flex flex-1 flex-col gap-2">
  <span class="text-sm font-medium">Spectrum Tokens</span>

  <PatchFrame busy={coverage.isPending || spectrum.isFetching || tokens.isFetching} class="bg-card">
    {#if spectrum.data && cells && palette}
      <SpectrumChart
        spectrum={spectrum.data}
        values={cells}
        color={palette}
        opacity={map ? undefined : tokenAlpha}
        title={caption}
        {selected}
        {onselect}
        label={onselect
          ? 'Scroll to zoom, drag to pan, click a span to select its token'
          : `Spectrum of galaxy ${galaxy}`}
        class={onselect ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
      />
    {:else if coverage.data && !matched}
      <p class="absolute inset-0 grid place-content-center text-xs text-muted-foreground">
        No spectrum
      </p>
    {/if}
  </PatchFrame>
</div>
