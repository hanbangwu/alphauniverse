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

  const STEPS: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1 }
</script>

<script lang="ts">
  import { type RGB, SELECTED, chartInk } from '$lib/color'
  import { TOOLTIP, chart } from '$lib/components/common/chart.svelte'
  import { type Spectrum, spanAt, spanOf } from '$lib/data/spectra'
  import { getMeta } from '$lib/state/app.svelte'
  import type { TooltipComponentFormatterCallbackParams } from 'echarts'
  import type { ECharts, ElementEvent } from 'echarts/core'
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
  let cursor: number | null = null

  const first = $derived(spectrum.wavelength[0])
  const last = $derived(spectrum.wavelength[spectrum.wavelength.length - 1])

  const points = $derived(
    Array.from(spectrum.wavelength, (wavelength, index) => [wavelength, spectrum.flux[index]])
  )

  const areas = $derived.by((): Areas => {
    if (!values || !color) return []
    const chosen = new Set(selected)
    return Array.from(values, (value, index): Areas[number] => {
      const [low, high] = spanOf(meta, index)
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
    const index = spanAt(meta, wavelength)
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
    dataZoom: [{ type: 'inside' }],
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

  function pick(event: ElementEvent): void {
    if (!values) return
    const position = [event.offsetX, event.offsetY]
    if (!instance.containPixel('grid', position)) return
    const index = spanAt(meta, instance.convertFromPixel('grid', position)[0])
    if (index < 0 || index >= values.length) return
    cursor = null
    ontoggle?.(index)
  }

  function shown(): [number, number] {
    const [{ startValue, endValue }] = instance.getOption().dataZoom as {
      startValue: number
      endValue: number
    }[]
    return [startValue, endValue]
  }

  function point(index: number | null): void {
    cursor = index
    if (index === null) {
      instance.dispatchAction({ type: 'showTip', x: -1, y: -1 })
      return
    }
    const [low, high] = spanOf(meta, index)
    const left = Math.max(low, first)
    const right = Math.min(high, last)
    const y = instance.getHeight() / 2
    const pixel = (value: number) => instance.convertToPixel({ xAxisIndex: 0 }, value)
    if (![left, right].every((value) => instance.containPixel('grid', [pixel(value), y]))) {
      const [from, to] = shown()
      const shift =
        right - left > to - from
          ? (left + right - from - to) / 2
          : Math.min(left - from, 0) + Math.max(right - to, 0)
      instance.dispatchAction({ type: 'dataZoom', startValue: from + shift, endValue: to + shift })
    }
    instance.dispatchAction({ type: 'showTip', x: pixel((left + right) / 2), y })
  }

  function move(event: KeyboardEvent): void {
    const step = STEPS[event.key]
    if (!step || event.altKey || event.ctrlKey || event.metaKey) return
    event.preventDefault()
    if (cursor === null) {
      const [from, to] = shown()
      point(spanAt(meta, (from + to) / 2))
      return
    }
    point(Math.min(Math.max(cursor + step, spanAt(meta, first)), spanAt(meta, last)))
  }

  function activate(event: MouseEvent): void {
    if (event.detail === 0 && cursor !== null) ontoggle?.(cursor)
  }
</script>

<svelte:element
  this={interactive ? 'button' : 'div'}
  {@attach chart(
    () => option,
    (created) => {
      instance = created
      created.getZr().on('click', pick)
    }
  )}
  onkeydown={move}
  onclick={activate}
  onblur={() => point(null)}
  type={interactive ? 'button' : undefined}
  class={className}
  role={interactive ? undefined : 'img'}
  aria-label={label}
></svelte:element>
