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

  const HOVER = { stroke: '#fff', lineWidth: 1 }
  const STEPS: Record<string, [number, number]> = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, -1],
    ArrowDown: [0, 1]
  }
</script>

<script lang="ts">
  import { type RGB, SELECTED } from '$lib/color'
  import { TOOLTIP, chart } from '$lib/components/common/chart.svelte'
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
  let cursor = $state<number | null>(null)

  const cells = $derived(Array.from(values, (_, index) => [index % grid, Math.floor(index / grid)]))

  function cell(
    params: CustomSeriesRenderItemParams,
    api: CustomSeriesRenderItemAPI
  ): CustomSeriesRenderItemReturn {
    const [x, y] = api.coord([api.value(0), api.value(1)])
    const [width, height] = api.size!([1, 1]) as number[]
    const [r, g, b] = color(values[params.dataIndex])
    const picked = selected.includes(params.dataIndex)
    const pointed = params.dataIndex === cursor
    return {
      type: 'rect',
      shape: { x, y, width, height },
      style: {
        fill: `rgba(${r}, ${g}, ${b}, ${opacity(picked)})`,
        stroke: picked ? SELECTED : pointed ? HOVER.stroke : undefined,
        lineWidth: picked ? 2 : HOVER.lineWidth
      },
      z2: picked || pointed ? 1 : 0,
      transition: 'style',
      emphasis: { style: picked ? {} : HOVER }
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

  function point(index: number | null): void {
    cursor = index
    instance.dispatchAction(
      index === null ? { type: 'hideTip' } : { type: 'showTip', seriesIndex: 0, dataIndex: index }
    )
  }

  function move(event: KeyboardEvent): void {
    const step = STEPS[event.key]
    if (!step || event.altKey || event.ctrlKey || event.metaKey) return
    event.preventDefault()
    if (cursor === null) {
      point(0)
      return
    }
    const [column, row] = cells[cursor]
    const clamp = (value: number) => Math.min(Math.max(value, 0), grid - 1)
    point(clamp(row + step[1]) * grid + clamp(column + step[0]))
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
      created.on('click', ({ dataIndex }) => {
        cursor = null
        ontoggle?.(dataIndex)
      })
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
