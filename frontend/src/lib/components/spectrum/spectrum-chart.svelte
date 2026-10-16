<script module lang="ts">
  import { LineChart, type LineSeriesOption } from 'echarts/charts'
  import {
    DataZoomInsideComponent,
    type DataZoomComponentOption,
    GridComponent,
    type GridComponentOption,
    MarkAreaComponent,
    type MarkAreaComponentOption,
    type TooltipComponentOption
  } from 'echarts/components'
  import { type ComposeOption, use } from 'echarts/core'

  use([LineChart, GridComponent, DataZoomInsideComponent, MarkAreaComponent])

  type Option = ComposeOption<
    LineSeriesOption | GridComponentOption | TooltipComponentOption | DataZoomComponentOption
  >
  type Areas = NonNullable<MarkAreaComponentOption['data']>
</script>

<script lang="ts">
  import { PREDICTED, type RGB, SELECTED, chartInk } from '$lib/color'
  import { TOOLTIP, chart } from '$lib/components/common/chart.svelte'
  import { type Spectrum, spectrumTokenAt, spectrumTokenOf } from '$lib/data/spectra'
  import { must } from '$lib/invariant'
  import { getMeta } from '$lib/state/app.svelte'
  import type { TooltipComponentFormatterCallbackParams } from 'echarts'
  import type { ECharts, ElementEvent } from 'echarts/core'
  import { mode } from 'mode-watcher'

  interface Props {
    spectrum: Spectrum | null
    prediction?: Spectrum | null
    values?: ArrayLike<number> | null
    color?: ((value: number) => RGB) | null
    opacity?: (picked: boolean) => number
    title?: (value: number, index: number) => string
    selected?: number[]
    ontoggle?: (index: number) => void
    class?: string
    label: string
  }

  let {
    spectrum,
    prediction = null,
    values,
    color,
    opacity = () => 1,
    title,
    selected = [],
    ontoggle,
    class: className,
    label
  }: Props = $props()

  const meta = getMeta()

  const interactive = $derived(Boolean(ontoggle))
  const ink = $derived(chartInk(mode.current ?? 'light'))

  function line(drawn: Spectrum | null): number[][] {
    return drawn
      ? Array.from(drawn.wavelength, (wavelength, index) => [wavelength, drawn.flux[index]])
      : []
  }

  const points = $derived(line(spectrum))
  const predicted = $derived(line(prediction))
  const wavelengths = $derived(
    must(spectrum ?? prediction, 'a spectrum or a prediction').wavelength
  )

  const areas = $derived.by((): Areas => {
    if (!values || !color) return []
    const chosen = new Set(selected)
    return Array.from(values, (value, index): Areas[number] => {
      const [low, high] = spectrumTokenOf(meta, index)
      const [r, g, b] = color(value)
      const picked = chosen.has(index)
      return [
        {
          xAxis: low,
          itemStyle: {
            color: `rgba(${r}, ${g}, ${b}, ${opacity(picked)})`,
            borderColor: SELECTED,
            borderWidth: picked ? 1 : 0
          }
        },
        { xAxis: high }
      ]
    })
  })

  function describe(params: TooltipComponentFormatterCallbackParams): string {
    const [wavelength, flux] = (Array.isArray(params) ? params[0] : params).value as [
      number,
      number
    ]
    const index = spectrumTokenAt(meta, wavelength)
    const value = values?.[index]
    return [
      `${wavelength.toFixed(1)} Å`,
      Number.isNaN(flux) ? 'masked' : flux.toFixed(2),
      ...(value === undefined || !title ? [] : [title(value, index)])
    ].join(' · ')
  }

  const option = $derived<Option>({
    animation: false,
    grid: { left: 8, right: 12, top: 8, bottom: 8 },
    xAxis: {
      type: 'value',
      min: Math.floor(wavelengths[0]),
      max: Math.ceil(wavelengths[wavelengths.length - 1]),
      name: 'Wavelength (Å)',
      nameLocation: 'middle',
      nameTextStyle: { color: ink.muted },
      axisLabel: { color: ink.muted },
      axisLine: { lineStyle: { color: ink.rule } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      scale: true,
      name: spectrum ? 'Flux (10^-17 erg/s/cm^2/Å)' : 'Normalised flux',
      nameLocation: 'middle',
      nameTextStyle: { color: ink.muted },
      axisLabel: { color: ink.muted },
      splitLine: { lineStyle: { color: ink.rule } }
    },
    tooltip: interactive
      ? {
          ...TOOLTIP,
          trigger: 'axis',
          formatter: describe,
          axisPointer: { lineStyle: { color: ink.muted } }
        }
      : undefined,
    dataZoom: [{ type: 'inside' }],
    series: [
      {
        type: 'line',
        data: points,
        showSymbol: false,
        lineStyle: { width: 1, color: ink.text },
        emphasis: { disabled: true },
        markArea: { silent: true, data: areas }
      },
      {
        type: 'line',
        data: predicted,
        showSymbol: false,
        lineStyle: { width: 1, color: PREDICTED },
        emphasis: { disabled: true }
      }
    ]
  })

  function pick(instance: ECharts, event: ElementEvent): void {
    if (!values) return
    const point = [event.offsetX, event.offsetY]
    if (!instance.containPixel('grid', point)) return
    const index = spectrumTokenAt(meta, instance.convertFromPixel('grid', point)[0])
    if (index < 0 || index >= values.length) return
    ontoggle?.(index)
  }
</script>

<svelte:element
  this={interactive ? 'button' : 'div'}
  {@attach chart(
    () => option,
    (instance) => instance.getZr().on('click', (event) => pick(instance, event))
  )}
  type={interactive ? 'button' : undefined}
  class={className}
  role={interactive ? undefined : 'img'}
  aria-label={label}
></svelte:element>

{#if prediction}
  <span
    class="pointer-events-none absolute top-1.5 right-3 flex items-center gap-1 rounded-sm bg-card/80 px-1 text-[10px] text-muted-foreground"
  >
    <span class="h-0.5 w-3" style:background-color={PREDICTED}></span>
    {spectrum ? 'AION' : 'AION prediction, normalised'}
  </span>
{/if}
