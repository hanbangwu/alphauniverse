<script module lang="ts">
  import { CustomChart, type CustomSeriesOption } from 'echarts/charts'
  import {
    GridComponent,
    type GridComponentOption,
    type TooltipComponentOption
  } from 'echarts/components'
  import { type ComposeOption, use } from 'echarts/core'

  use([CustomChart, GridComponent])

  type Option = ComposeOption<CustomSeriesOption | GridComponentOption | TooltipComponentOption>
</script>

<script lang="ts">
  import { type RGB, SELECTED } from '$lib/color'
  import { TOOLTIP, chart, pixel } from '$lib/components/common/chart.svelte'
  import type {
    CustomSeriesRenderItemAPI,
    CustomSeriesRenderItemParams,
    CustomSeriesRenderItemReturn
  } from 'echarts'
  import type { ECharts } from 'echarts/core'

  interface Props {
    values: ArrayLike<number>
    grid: number
    color: (value: number) => RGB
    opacity?: (picked: boolean) => number
    title?: (value: number, index: number) => string
    selected?: number[]
    ontoggle?: (index: number) => void
    class?: string
    label: string
  }

  let {
    values,
    grid,
    color,
    opacity = () => 1,
    title,
    selected = [],
    ontoggle,
    class: className,
    label
  }: Props = $props()

  const interactive = $derived(Boolean(ontoggle))

  let instance: ECharts
  let painted = $state<number[]>([])
  let adding = false
  let last: number[] | null = null

  const cells = $derived(Array.from(values, (_, index) => [index % grid, Math.floor(index / grid)]))

  function cell(
    params: CustomSeriesRenderItemParams,
    api: CustomSeriesRenderItemAPI
  ): CustomSeriesRenderItemReturn {
    const [x, y] = api.coord([api.value(0), api.value(1)])
    const [width, height] = api.size!([1, 1]) as number[]
    const [r, g, b] = color(values[params.dataIndex])
    const picked = selected.includes(params.dataIndex) !== painted.includes(params.dataIndex)
    return {
      type: 'rect',
      shape: { x, y, width, height },
      style: {
        fill: `rgba(${r}, ${g}, ${b}, ${opacity(picked)})`,
        stroke: picked ? SELECTED : undefined,
        lineWidth: 2
      },
      z2: picked ? 1 : 0,
      transition: 'style',
      emphasis: { style: picked ? {} : { stroke: '#fff', lineWidth: 1 } }
    }
  }

  const option = $derived<Option>({
    animationDurationUpdate: 150,
    grid: { left: 0, right: 0, top: 0, bottom: 0 },
    xAxis: { type: 'value', min: 0, max: grid, show: false },
    yAxis: { type: 'value', min: 0, max: grid, inverse: true, show: false },
    tooltip: title
      ? {
          ...TOOLTIP,
          trigger: 'item',
          formatter: (params) => {
            const { dataIndex } = Array.isArray(params) ? params[0] : params
            return title(values[dataIndex], dataIndex)
          }
        }
      : undefined,
    series: [{ type: 'custom', data: cells, renderItem: cell }]
  })

  function position(event: PointerEvent): number[] {
    return instance.convertFromPixel('grid', pixel(instance, event))
  }

  function paint([column, row]: number[]): void {
    const index = row * grid + column
    if (Math.min(column, row) < 0 || Math.max(column, row) >= grid) return
    if (selected.includes(index) !== adding && !painted.includes(index)) painted.push(index)
  }

  function start(event: PointerEvent): void {
    if (last) {
      end(event)
      return
    }
    if (event.button !== 0) return
    last = position(event)
    const [column, row] = last.map(Math.floor)
    adding = !selected.includes(row * grid + column)
    paint([column, row])
  }

  function move(event: PointerEvent): void {
    if (!last) return
    if (!event.buttons) {
      end(event)
      return
    }
    const from = last
    const to = (last = position(event))
    const steps = Math.ceil(Math.max(...from.map((value, axis) => Math.abs(to[axis] - value))))
    for (let step = 1; step <= steps; step++) {
      paint(from.map((value, axis) => Math.floor(value + ((to[axis] - value) * step) / steps)))
    }
  }

  function end({ type }: PointerEvent): void {
    if (!last) return
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
  class={[className, interactive && 'touch-pinch-zoom']}
  role={interactive ? undefined : 'img'}
  aria-label={label}
></svelte:element>
