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
  import type { RGB } from '$lib/color'
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
    painted: number[]
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
    painted = $bindable(),
    ontoggle,
    class: className,
    label
  }: Props = $props()

  const interactive = $derived(Boolean(ontoggle))

  let instance: ECharts
  let adding: boolean
  let pointer: number | null = null
  let last: number[]

  const cells = $derived(Array.from(values, (_, index) => [index % grid, Math.floor(index / grid)]))
  const chosen = $derived(new Set(selected).symmetricDifference(new Set(painted)))

  function cell(
    params: CustomSeriesRenderItemParams,
    api: CustomSeriesRenderItemAPI
  ): CustomSeriesRenderItemReturn {
    const [x, y] = api.coord([api.value(0), api.value(1)])
    const [width, height] = api.size!([1, 1]) as number[]
    const [r, g, b] = color(values[params.dataIndex])
    return {
      type: 'rect',
      shape: { x, y, width, height },
      style: { fill: `rgba(${r}, ${g}, ${b}, ${opacity(chosen.has(params.dataIndex))})` },
      transition: 'style',
      emphasis: { style: { stroke: '#fff', lineWidth: 1 } }
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
    if (chosen.has(index) !== adding) painted = [...painted, index]
  }

  function start(event: PointerEvent): void {
    if (pointer !== null) {
      drop()
      return
    }
    if (event.button !== 0) return
    pointer = event.pointerId
    last = position(event)
    const [column, row] = last.map(Math.floor)
    adding = !selected.includes(row * grid + column)
    paint([column, row])
  }

  function move(event: PointerEvent): void {
    if (event.pointerId !== pointer) return
    if (!event.buttons) {
      drop()
      return
    }
    const [fromX, fromY] = last
    last = position(event)
    const [toX, toY] = last
    let [column, row] = [Math.floor(fromX), Math.floor(fromY)]
    const [endColumn, endRow] = [Math.floor(toX), Math.floor(toY)]
    while (column !== endColumn || row !== endRow) {
      const across = (column + Number(toX > fromX) - fromX) / (toX - fromX)
      const down = (row + Number(toY > fromY) - fromY) / (toY - fromY)
      if (row === endRow || (column !== endColumn && across < down)) {
        column += Math.sign(toX - fromX)
      } else {
        row += Math.sign(toY - fromY)
      }
      paint([column, row])
    }
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
  onpointerdown={interactive ? start : undefined}
  type={interactive ? 'button' : undefined}
  class={[className, interactive && 'touch-pan-y touch-pinch-zoom']}
  role={interactive ? undefined : 'img'}
  aria-label={label}
></svelte:element>
