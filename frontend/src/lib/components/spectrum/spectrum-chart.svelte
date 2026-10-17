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
  import { OUTLINE, PREDICTED, type RGB, chartInk } from '$lib/color'
  import { TAP_RADIUS, TOOLTIP, chart, pixel } from '$lib/components/common/chart.svelte'
  import { type Spectrum, spectrumTokenAt, spectrumTokenOf } from '$lib/data/spectra'
  import { must } from '$lib/invariant'
  import { getMeta } from '$lib/state/app.svelte'
  import type { TooltipComponentFormatterCallbackParams } from 'echarts'
  import type { ECharts } from 'echarts/core'
  import { mode } from 'mode-watcher'

  interface Props {
    spectrum: Spectrum | null
    prediction?: Spectrum | null
    values?: ArrayLike<number> | null
    color?: ((value: number) => RGB) | null
    mask?: ArrayLike<number> | null
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
    mask,
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
  let cursor: number | null = null

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
  const first = $derived(wavelengths[0])
  const last = $derived(wavelengths[wavelengths.length - 1])

  const chosen = $derived(new Set(selected).symmetricDifference(new Set(painted)))

  const areas = $derived.by((): Areas => {
    if (!values || !color) return []
    const cells = Array.from(values, (value, index): Areas[number] => {
      const [low, high] = spectrumTokenOf(meta, index)
      const [r, g, b] = color(value)
      return [
        {
          xAxis: low,
          itemStyle: {
            color: `rgba(${r}, ${g}, ${b}, ${mask?.[index] === 0 ? 0 : opacity(chosen.has(index))})`
          }
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
      min: Math.floor(first),
      max: Math.ceil(last),
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
    dataZoom: [{ type: 'inside', cursorGrab: interactive ? 'pointer' : 'default' }],
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
    cursor = null
    visible = shown()
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
    if (event.pointerType !== 'mouse') {
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
    const [low, high] = spectrumTokenOf(meta, index)
    const left = Math.max(low, first)
    const right = Math.min(high, last)
    const y = instance.getHeight() / 2
    const toPixel = (value: number) => instance.convertToPixel({ xAxisIndex: 0 }, value)
    if (![left, right].every((value) => instance.containPixel('grid', [toPixel(value), y]))) {
      const [from, to] = shown()
      const shift =
        right - left > to - from
          ? (left + right - from - to) / 2
          : Math.min(left - from, 0) + Math.max(right - to, 0)
      instance.dispatchAction({ type: 'dataZoom', startValue: from + shift, endValue: to + shift })
    }
    instance.dispatchAction({ type: 'showTip', x: toPixel((left + right) / 2), y })
  }

  function navigate(event: KeyboardEvent): void {
    const step = STEPS[event.key]
    if (!step || event.altKey || event.ctrlKey || event.metaKey) return
    event.preventDefault()
    if (cursor === null) {
      const [from, to] = shown()
      point(spectrumTokenAt(meta, (from + to) / 2))
      return
    }
    point(
      Math.min(Math.max(cursor + step, spectrumTokenAt(meta, first)), spectrumTokenAt(meta, last))
    )
  }

  function activate(event: MouseEvent): void {
    if (event.detail === 0 && cursor !== null) ontoggle?.(cursor)
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
  onkeydown={navigate}
  onclick={activate}
  onblur={() => point(null)}
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
