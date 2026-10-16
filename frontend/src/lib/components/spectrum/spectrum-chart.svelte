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
  import { type RGB, SELECTED, chartInk } from '$lib/color'
  import { TOOLTIP, chart } from '$lib/components/common/chart.svelte'
  import { type Spectrum, spectrumTokenAt, spectrumTokenOf } from '$lib/data/spectra'
  import { getMeta } from '$lib/state/app.svelte'
  import type { TooltipComponentFormatterCallbackParams } from 'echarts'
  import type { ECharts } from 'echarts/core'
  import { mode } from 'mode-watcher'

  interface Props {
    spectrum: Spectrum
    values?: ArrayLike<number>
    color?: (value: number) => RGB
    opacity?: (picked: boolean) => number
    title?: (value: number, index: number) => string
    selected?: number[]
    ontoggle?: (index: number) => void
    class?: string
    label: string
  }

  let {
    spectrum,
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

  let instance: ECharts
  let painted = $state<number[]>([])
  let adding = false
  let last: number | null = null

  const points = $derived(
    Array.from(spectrum.wavelength, (wavelength, index) => [wavelength, spectrum.flux[index]])
  )

  const areas = $derived.by((): Areas => {
    if (!values || !color) return []
    const chosen = new Set(selected)
    return Array.from(values, (value, index): Areas[number] => {
      const [low, high] = spectrumTokenOf(meta, index)
      const [r, g, b] = color(value)
      const picked = chosen.has(index) !== painted.includes(index)
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
      min: 'dataMin',
      max: 'dataMax',
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
      name: 'Flux (10^-17 erg/s/cm^2/Å)',
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
    dataZoom: [
      interactive
        ? {
            type: 'inside',
            moveOnMouseMove: 'shift',
            cursorGrab: 'pointer',
            cursorGrabbing: 'pointer'
          }
        : { type: 'inside' }
    ],
    series: [
      {
        type: 'line',
        data: points,
        showSymbol: false,
        lineStyle: { width: 1, color: ink.text },
        emphasis: { disabled: true },
        markArea: { silent: true, data: areas }
      }
    ]
  })

  function pixel({ clientX, clientY }: PointerEvent): number[] {
    const { left, top } = instance.getDom().getBoundingClientRect()
    return [clientX - left, clientY - top]
  }

  function windowAt(event: PointerEvent): number {
    const [{ startValue, endValue }] = instance.getOption().dataZoom as {
      startValue: number
      endValue: number
    }[]
    const [wavelength] = instance.convertFromPixel('grid', pixel(event))
    return spectrumTokenAt(meta, Math.min(Math.max(wavelength, startValue), endValue))
  }

  function paint(index: number): void {
    if (selected.includes(index) !== adding && !painted.includes(index)) painted.push(index)
  }

  function start(event: PointerEvent): void {
    if (last !== null) {
      end(event)
      return
    }
    if (event.button !== 0 || event.shiftKey || !instance.containPixel('grid', pixel(event))) return
    last = windowAt(event)
    adding = !selected.includes(last)
    paint(last)
  }

  function move(event: PointerEvent): void {
    if (last === null) return
    if (!event.buttons) {
      end(event)
      return
    }
    const index = windowAt(event)
    for (let between = Math.min(last, index); between <= Math.max(last, index); between++) {
      paint(between)
    }
    last = index
  }

  function end({ type }: PointerEvent): void {
    if (last === null) return
    if (type === 'pointerup') painted.forEach((index) => ontoggle?.(index))
    painted = []
    last = null
  }
</script>

<svelte:window onpointermove={move} onpointerup={end} onpointercancel={end} />

<svelte:element
  this={interactive ? 'button' : 'div'}
  {@attach chart(
    () => option,
    (created) => (instance = created)
  )}
  onpointerdown={interactive ? start : undefined}
  type={interactive ? 'button' : undefined}
  class={className}
  role={interactive ? undefined : 'img'}
  aria-label={label}
></svelte:element>
