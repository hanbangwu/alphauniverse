<script lang="ts">
  import type { SpectrumSurvey } from '$lib/client'
  import { type RGB, tokenAlpha, tokenColors } from '$lib/color'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import { spectrumQuery, spectrumTemplateQuery, spectrumTokensQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { SURVEYS } from '$lib/labels'
  import { getView } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    survey: SpectrumSurvey
    matched: boolean
    map: Float32Array | null
    selected?: number[]
    ontoggle?: (index: number) => void
    heat: ((value: number) => RGB) | null
    mask?: ArrayLike<number> | null
    caption: (value: number, index: number) => string
    error?: string | null
  }

  let { galaxy, survey, matched, map, selected, ontoggle, heat, mask, caption, error }: Props =
    $props()

  const similarity = getSimilarity()
  const view = getView()

  const spectrum = createQuery(() => spectrumQuery(matched ? galaxy : null, survey))
  const tokens = createQuery(() => spectrumTokensQuery(matched && !map ? galaxy : null, survey))
  const prediction = createQuery(() =>
    spectrumTemplateQuery(view.predictions.value ? galaxy : null, survey)
  )

  const cells = $derived(map ?? tokens.data ?? null)
  const palette = $derived(map ? heat : tokens.data ? tokenColors(tokens.data) : null)
  const picking = $derived(matched ? ontoggle : undefined)
  const label = $derived(SURVEYS[survey].label)
  const message = $derived(
    spectrum.isError
      ? errorMessage(spectrum.error)
      : prediction.isError
        ? errorMessage(prediction.error)
        : error
  )
</script>

<div class="flex flex-col gap-2">
  <span class="text-sm font-medium">{label}</span>

  <PanelFrame
    busy={spectrum.isFetching || tokens.isFetching || prediction.isFetching}
    class="aspect-2/1 bg-card md:aspect-5/1"
  >
    {#if matched ? spectrum.data && cells && palette : prediction.data}
      <SpectrumChart
        spectrum={spectrum.data ?? null}
        prediction={prediction.data}
        values={cells}
        color={palette}
        {mask}
        opacity={map ? undefined : tokenAlpha}
        title={map ? caption : similarity.caption}
        {selected}
        ontoggle={picking}
        label={picking
          ? `Scroll to zoom, Shift-drag to pan, click or drag across ${label} spectrum tokens to select them`
          : matched
            ? `${label} spectrum of galaxy ${galaxy}`
            : `AION's predicted ${label} spectrum of galaxy ${galaxy}, normalised`}
        class="absolute inset-0"
      />
    {/if}
  </PanelFrame>
  {#if message}
    <p role="alert" class="text-xs leading-relaxed text-destructive">{message}</p>
  {/if}
</div>
