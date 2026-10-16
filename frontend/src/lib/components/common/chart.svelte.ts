import { TooltipComponent, type TooltipComponentOption } from 'echarts/components'
import { type ECharts, type EChartsCoreOption, init, use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import type { Attachment } from 'svelte/attachments'

use([TooltipComponent, CanvasRenderer])

export const TOOLTIP: TooltipComponentOption = {
  confine: true,
  backgroundColor: 'rgba(0, 0, 0, 0.75)',
  borderWidth: 0,
  padding: [2, 6],
  textStyle: { color: '#fff', fontFamily: 'monospace', fontSize: 10 }
}

export function pixel(instance: ECharts, { clientX, clientY }: PointerEvent): number[] {
  const { left, top } = instance.getDom().getBoundingClientRect()
  return [clientX - left, clientY - top]
}

export function chart(
  option: () => EChartsCoreOption,
  oncreate: (instance: ECharts) => void
): Attachment<HTMLElement> {
  return (element) => {
    const instance = init(element)
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(element)
    oncreate(instance)
    $effect(() => {
      instance.setOption(option())
    })
    return () => {
      observer.disconnect()
      instance.dispose()
    }
  }
}
