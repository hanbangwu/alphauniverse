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

  const TAP_RADIUS = 10
</script>

<script lang="ts">
  import { OUTLINE, type RGB, chartInk } from '$lib/color'
  import { TOOLTIP, chart, pixel } from '$lib/components/common/chart.svelte'
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
  let painted = $state.raw<number[]>([])
  let adding: boolean
  let pointer: number | null = null
  let previous: number
  let visible: number[]
  let press: number[]

  const points = $derived(
    Array.from(spectrum.wavelength, (wavelength, index) => [wavelength, spectrum.flux[index]])
  )

  const chosen = $derived(new Set(selected).symmetricDifference(new Set(painted)))

  const areas = $derived.by((): Areas => {
    if (!values || !color) return []
    const cells = Array.from(values, (value, index): Areas[number] => {
      const [low, high] = spectrumTokenOf(meta, index)
      const [r, g, b] = color(value)
      return [
        {
          xAxis: low,
          itemStyle: { color: `rgba(${r}, ${g}, ${b}, ${opacity(chosen.has(index))})` }
        },
        { xAxis: high }
      ]
    })
    const runs: [number, number][] = []
    for (const index of [...chosen].sort((a, b) => a - b)) {
      const run = runs.at(-1)
      if (run && index === run[1] + 1) run[1] = index
      else runs.push([index, index])
    }
    const outlines = runs.flatMap(([first, last]) =>
      OUTLINE.map(({ stroke, width }, layer): Areas[number] => [
        {
          xAxis: spectrumTokenOf(meta, first)[0],
          itemStyle: { color: 'transparent', borderColor: stroke, borderWidth: width },
          z2: layer + 1
        },
        { xAxis: spectrumTokenOf(meta, last)[1] }
      ])
    )
    return [...cells, ...outlines]
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
    dataZoom: [{ type: 'inside', cursorGrab: interactive ? 'pointer' : 'default' }],
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

  function windowAt(event: PointerEvent): number {
    const [wavelength] = instance.convertFromPixel('grid', pixel(instance, event))
    const [low, high] = visible
    return spectrumTokenAt(meta, Math.min(Math.max(wavelength, low), high))
  }

  function paint(index: number): void {
    if (chosen.has(index) !== adding) painted = [...painted, index]
  }

  function blockPan(event: MouseEvent): void {
    if (!event.shiftKey) event.stopPropagation()
  }

  function intercept(event: PointerEvent): void {
    if (event.pointerType !== 'touch') blockPan(event)
    if (interactive && !event.shiftKey) start(event)
  }

  function start(event: PointerEvent): void {
    if (pointer !== null) {
      drop()
      return
    }
    if (event.button !== 0) return
    if (!instance.containPixel('grid', pixel(instance, event))) return
    const [{ startValue, endValue }] = instance.getOption().dataZoom as {
      startValue: number
      endValue: number
    }[]
    visible = [startValue, endValue]
    pointer = event.pointerId
    press = [event.clientX, event.clientY]
    previous = windowAt(event)
    adding = !selected.includes(previous)
    paint(previous)
  }

  function move(event: PointerEvent): void {
    if (event.pointerId !== pointer) return
    if (!event.buttons) {
      drop()
      return
    }
    if (event.pointerType === 'touch') {
      if (Math.hypot(event.clientX - press[0], event.clientY - press[1]) > TAP_RADIUS) drop()
      return
    }
    const index = windowAt(event)
    for (let between = Math.min(previous, index); between <= Math.max(previous, index); between++) {
      paint(between)
    }
    previous = index
  }

  function end(event: PointerEvent): void {
    if (event.pointerId !== pointer) return
    if (event.type === 'pointerup') painted.forEach((index) => ontoggle?.(index))
    drop()
  }

  function drop(): void {
    painted = []
    pointer = null
  }
</script>

<svelte:window onpointermove={move} onpointerup={end} onpointercancel={end} />

<svelte:element
  this={interactive ? 'button' : 'div'}
  {@attach chart(
    () => option,
    (created) => (instance = created)
  )}
  onpointerdowncapture={intercept}
  onmousedowncapture={blockPan}
  type={interactive ? 'button' : undefined}
  class={className}
  role={interactive ? undefined : 'img'}
  aria-label={label}
></svelte:element>
