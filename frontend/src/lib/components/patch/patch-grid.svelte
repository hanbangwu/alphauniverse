<script module lang="ts">
  import { CustomChart, type CustomSeriesOption } from 'echarts/charts'
  import {
    GridComponent,
    type GridComponentOption,
    TooltipComponent,
    type TooltipComponentOption
  } from 'echarts/components'
  import { type ComposeOption, use } from 'echarts/core'
  import { CanvasRenderer } from 'echarts/renderers'

  use([CustomChart, GridComponent, TooltipComponent, CanvasRenderer])

  type Option = ComposeOption<CustomSeriesOption | GridComponentOption | TooltipComponentOption>
</script>

<script lang="ts">
  import { type RGB, SELECTED } from '$lib/color'
  import { TOOLTIP, chart } from '$lib/components/common/chart.svelte'
  import type {
    CustomSeriesRenderItemAPI,
    CustomSeriesRenderItemParams,
    CustomSeriesRenderItemReturn
  } from 'echarts'

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

  const cells = $derived(Array.from(values, (_, index) => [index % grid, Math.floor(index / grid)]))

  function cell(
    params: CustomSeriesRenderItemParams,
    api: CustomSeriesRenderItemAPI
  ): CustomSeriesRenderItemReturn {
    const [x, y] = api.coord([api.value(0), api.value(1)])
    const [width, height] = api.size!([1, 1]) as number[]
    const [r, g, b] = color(values[params.dataIndex])
    const picked = selected.includes(params.dataIndex)
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
</script>

<svelte:element
  this={interactive ? 'button' : 'div'}
  {@attach chart(
    () => option,
    (instance) => instance.on('click', ({ dataIndex }) => ontoggle?.(dataIndex))
  )}
  type={interactive ? 'button' : undefined}
  class={className}
  role={interactive ? undefined : 'img'}
  aria-label={label}
></svelte:element>
