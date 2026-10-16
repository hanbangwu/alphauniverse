<script lang="ts">
  import { type RGB, tokenAlpha, tokenColors } from '$lib/color'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import {
    coverageQuery,
    spectrumQuery,
    spectrumTemplateQuery,
    spectrumTokensQuery
  } from '$lib/data/queries'
  import { hasSpectrum } from '$lib/data/spectra'
  import { errorMessage } from '$lib/errors'
  import { getView } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    map: Float32Array | null
    selected?: number[]
    ontoggle?: (index: number) => void
    absent?: string
    heat: ((value: number) => RGB) | null
    caption: (value: number, index: number) => string
  }

  let { galaxy, map, selected, ontoggle, absent = 'No spectrum', heat, caption }: Props = $props()

  const similarity = getSimilarity()
  const view = getView()

  const coverage = createQuery(() => coverageQuery(galaxy))
  const matched = $derived(coverage.data ? hasSpectrum(coverage.data) : false)
  const spectrum = createQuery(() => spectrumQuery(matched ? galaxy : null))
  const tokens = createQuery(() => spectrumTokensQuery(matched && !map ? galaxy : null))
  const prediction = createQuery(() =>
    spectrumTemplateQuery(view.predictions.value ? galaxy : null)
  )

  const cells = $derived(map ?? tokens.data ?? null)
  const palette = $derived(map ? heat : tokens.data ? tokenColors(tokens.data) : null)
  const observed = $derived(matched ? (spectrum.data ?? null) : null)
  const predicted = $derived(prediction.data ?? null)
  const picking = $derived(matched ? ontoggle : undefined)
</script>

<div class="flex flex-col gap-2">
  <span class="text-sm font-medium">Spectrum Tokens</span>

  <PanelFrame
    busy={coverage.isPending || spectrum.isFetching || tokens.isFetching || prediction.isFetching}
    class="aspect-2/1 bg-card md:aspect-4/1"
  >
    {#if matched ? observed && cells && palette : predicted}
      <SpectrumChart
        spectrum={observed}
        prediction={predicted}
        values={cells}
        color={palette}
        opacity={map ? undefined : tokenAlpha}
        title={map ? caption : similarity.caption}
        {selected}
        ontoggle={picking}
        label={picking
          ? 'Scroll to zoom, drag to pan, click a spectrum token to select it'
          : observed
            ? `Spectrum of galaxy ${galaxy}`
            : `AION's predicted spectrum of galaxy ${galaxy}, normalised`}
        class={picking ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
      />
    {:else if coverage.data && !matched && !view.predictions.value}
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
  {#if prediction.isError}
    <p role="alert" class="text-xs leading-relaxed text-destructive">
      {errorMessage(prediction.error)}
    </p>
  {/if}
</div>
