import type { TooltipComponentOption } from 'echarts/components'
import { type ECharts, type EChartsCoreOption, init } from 'echarts/core'
import type { Attachment } from 'svelte/attachments'

export const TOOLTIP: TooltipComponentOption = {
  confine: true,
  backgroundColor: 'rgba(0, 0, 0, 0.75)',
  borderWidth: 0,
  padding: [2, 6],
  textStyle: { color: '#fff', fontFamily: 'monospace', fontSize: 10 }
}

export function chart(
  option: () => EChartsCoreOption,
  listen: (instance: ECharts) => void
): Attachment<HTMLElement> {
  return (element) => {
    const instance = init(element)
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(element)
    listen(instance)
    $effect(() => {
      instance.setOption(option())
    })
    return () => {
      observer.disconnect()
      instance.dispose()
    }
  }
}
