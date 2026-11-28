<script module lang="ts">
  import type { RGB } from '$lib/color'
  import { SPECTRUM_PATCHES } from '$lib/data/spectra'

  const PATCHES = Array.from({ length: SPECTRUM_PATCHES }, (_, index) => index)
  const BLACK = (): RGB => [0, 0, 0]
  const CLEAR = (): number => 0
  const HALF = (): number => 0.5
</script>

<script lang="ts">
  import type { GalaxyId, SpectrumSurvey } from '$lib/client'
  import { falseColors } from '$lib/color'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import SpectrumChart from '$lib/components/spectrum/spectrum-chart.svelte'
  import {
    spectrumQuery,
    spectrumAionPredictionQuery,
    spectrumFalseColorQuery
  } from '$lib/data/queries'
  import { SURVEYS } from '$lib/labels'
  import { getView } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: GalaxyId
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

  const predicted = $derived(view.dataSource.value === 'prediction')
  const spectrum = createQuery(() => spectrumQuery(matched && !predicted ? galaxy : null, survey))
  const prediction = createQuery(() =>
    spectrumAionPredictionQuery(predicted ? galaxy : null, survey)
  )
  const colors = createQuery(() =>
    spectrumFalseColorQuery(
      !map && similarity.spectrumFalseColor.value ? galaxy : null,
      survey,
      predicted
    )
  )
  const drawn = $derived(predicted ? prediction.data : spectrum.data)

  const palette = $derived(map ? heat : colors.data ? falseColors(colors.data) : null)
  const label = $derived(SURVEYS[survey])
  const message = $derived(
    spectrum.isError
      ? spectrum.error.message
      : prediction.isError
        ? prediction.error.message
        : colors.isError
          ? colors.error.message
          : error
  )
</script>

<div class="flex flex-col gap-2">
  <PanelFrame
    busy={spectrum.isFetching || prediction.isFetching || colors.isFetching}
    class="aspect-2/1 bg-card md:aspect-5/1"
  >
    {#if drawn}
      <SpectrumChart
        spectrum={drawn}
        normalized={predicted && !matched}
        values={map ?? PATCHES}
        color={palette ?? BLACK}
        {mask}
        opacity={map ? undefined : palette ? HALF : CLEAR}
        title={map ? caption : undefined}
        {selected}
        {ontoggle}
        label={ontoggle
          ? `Scroll to zoom, Shift-drag to pan, click or drag across ${label} spectrum patches to select them`
          : predicted
            ? `AION's predicted ${label} spectrum of galaxy ${galaxy}${matched ? '' : ', normalized'}`
            : `${label} spectrum of galaxy ${galaxy}`}
        class="absolute inset-0"
      />
    {/if}
  </PanelFrame>
  {#if message}
    <p role="alert" class="text-xs leading-relaxed text-destructive">{message}</p>
  {/if}
</div>
