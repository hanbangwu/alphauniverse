<script module lang="ts">
  import { LineChart, type LineSeriesOption } from 'echarts/charts'
  import {
    DataZoomInsideComponent,
    type DataZoomComponentOption,
    GridComponent,
    type GridComponentOption,
    MarkAreaComponent,
    type MarkAreaComponentOption,
    TooltipComponent,
    type TooltipComponentOption
  } from 'echarts/components'
  import { type ComposeOption, use } from 'echarts/core'
  import { CanvasRenderer } from 'echarts/renderers'

  use([
    LineChart,
    GridComponent,
    TooltipComponent,
    DataZoomInsideComponent,
    MarkAreaComponent,
    CanvasRenderer
  ])

  type Option = ComposeOption<
    LineSeriesOption | GridComponentOption | TooltipComponentOption | DataZoomComponentOption
  >
  type Areas = NonNullable<MarkAreaComponentOption['data']>
</script>

<script lang="ts">
  import { type RGB, SELECTED, chartInk } from '$lib/color'
  import { type Spectrum, spanAt, spanOf } from '$lib/data/spectra'
  import { getMeta } from '$lib/state/app.svelte'
  import type { TooltipComponentFormatterCallbackParams } from 'echarts'
  import { type ECharts, type ElementEvent, init } from 'echarts/core'
  import { mode } from 'mode-watcher'

  interface Props {
    spectrum: Spectrum
    tokens?: Uint32Array
    color?: (value: number) => RGB
    selected?: number[]
    onselect?: (indices: number[]) => void
    class?: string
    label: string
  }

  let {
    spectrum,
    tokens,
    color,
    selected = [],
    onselect,
    class: className,
    label
  }: Props = $props()

  const grid = getMeta().spectrum

  let box = $state<HTMLElement | null>(null)
  let chart = $state.raw<ECharts | null>(null)

  const interactive = $derived(Boolean(onselect))
  const ink = $derived(chartInk(mode.current ?? 'light'))

  const points = $derived(
    Array.from(spectrum.wavelength, (wavelength, index) => [wavelength, spectrum.flux[index]])
  )

  const areas = $derived.by((): Areas => {
    if (!tokens || !color) return []
    const chosen = new Set(selected)
    return Array.from(tokens, (token, index): Areas[number] => {
      const [low, high] = spanOf(grid, index)
      const [r, g, b] = color(token)
      const picked = chosen.has(index)
      return [
        {
          xAxis: low,
          itemStyle: {
            color: `rgba(${r}, ${g}, ${b}, ${picked ? 0.6 : 0.2})`,
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
    const token = tokens?.[spanAt(grid, wavelength)]
    return [
      `${wavelength.toFixed(1)} Å`,
      Number.isNaN(flux) ? 'masked' : flux.toFixed(2),
      ...(token === undefined ? [] : [`token ${token}`])
    ].join(' · ')
  }

  // Every selection change re-sets the option, so animating thousands of
  // points would cost a frame each time for nothing visible.
  const option = $derived<Option>({
    animation: false,
    grid: { left: 8, right: 12, top: 8, bottom: 8, containLabel: true },
    xAxis: {
      type: 'value',
      min: 'dataMin',
      max: 'dataMax',
      axisLabel: { color: ink.muted },
      axisLine: { lineStyle: { color: ink.rule } },
      axisTick: { show: false }
    },
    yAxis: {
      type: 'value',
      scale: true,
      axisLabel: { color: ink.muted },
      splitLine: { lineStyle: { color: ink.rule } }
    },
    tooltip: interactive
      ? {
          trigger: 'axis',
          confine: true,
          formatter: describe,
          backgroundColor: 'rgba(0, 0, 0, 0.75)',
          borderWidth: 0,
          padding: [2, 6],
          textStyle: { color: '#fff', fontFamily: 'monospace', fontSize: 10 },
          axisPointer: { lineStyle: { color: ink.muted } }
        }
      : undefined,
    dataZoom: interactive ? [{ type: 'inside' }] : undefined,
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

  $effect(() => {
    if (!box) return
    const instance = init(box)
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(box)
    instance.getZr().on('click', pick)
    chart = instance
    return () => {
      observer.disconnect()
      instance.dispose()
      chart = null
    }
  })

  $effect(() => {
    chart?.setOption(option)
  })

  function pick(event: ElementEvent): void {
    if (!chart || !tokens) return
    const point = [event.offsetX, event.offsetY]
    if (!chart.containPixel('grid', point)) return
    const index = spanAt(grid, chart.convertFromPixel('grid', point)[0])
    if (index < 0 || index >= tokens.length) return
    onselect?.(
      selected.includes(index) ? selected.filter((value) => value !== index) : [...selected, index]
    )
  }
</script>

<svelte:element
  this={interactive ? 'button' : 'div'}
  bind:this={box}
  type={interactive ? 'button' : undefined}
  class={className}
  role={interactive ? undefined : 'img'}
  aria-label={label}
></svelte:element>
