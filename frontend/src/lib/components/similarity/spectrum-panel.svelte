<script lang="ts">
  import type { SpectrumSurvey } from '$lib/client'
  import { type RGB, tokenAlpha, tokenColors } from '$lib/color'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { spectrumQuery, spectrumTokensQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { SURVEYS } from '$lib/labels'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    survey: SpectrumSurvey
    map: Float32Array | null
    selected?: number[]
    ontoggle?: (index: number) => void
    heat: ((value: number) => RGB) | null
    mask?: ArrayLike<number> | null
    caption: (value: number, index: number) => string
    error?: string | null
  }

  let { galaxy, survey, map, selected, ontoggle, heat, mask, caption, error }: Props = $props()

  const similarity = getSimilarity()

  const spectrum = createQuery(() => spectrumQuery(galaxy, survey))
  const tokens = createQuery(() => spectrumTokensQuery(map ? null : galaxy, survey))

  const cells = $derived(map ?? tokens.data ?? null)
  const palette = $derived(map ? heat : tokens.data ? tokenColors(tokens.data) : null)
  const label = $derived(SURVEYS[survey].label)
  const message = $derived(spectrum.isError ? errorMessage(spectrum.error) : error)
</script>

<div class="flex flex-col gap-2">
  <span class="text-sm font-medium">{label}</span>

  <PanelFrame
    busy={spectrum.isFetching || tokens.isFetching}
    class="aspect-2/1 bg-card md:aspect-5/1"
  >
    {#if spectrum.data && cells && palette}
      <SpectrumChart
        spectrum={spectrum.data}
        values={cells}
        color={palette}
        {mask}
        opacity={map ? undefined : tokenAlpha}
        title={map ? caption : similarity.caption}
        {selected}
        {ontoggle}
        label={ontoggle
          ? `Scroll to zoom, drag to pan, click a ${label} spectrum token to select it`
          : `${label} spectrum of galaxy ${galaxy}`}
        class={ontoggle ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
      />
    {/if}
  </PanelFrame>
  {#if message}
    <p role="alert" class="text-xs leading-relaxed text-destructive">{message}</p>
  {/if}
</div>
