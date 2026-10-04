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

export function chart(
  option: () => EChartsCoreOption,
  listen: (instance: ECharts) => void
): Attachment<HTMLElement> {
  return (element) => {
    let visible = $state(false)
    let instance: ECharts | null = null
    let applied: EChartsCoreOption | null = null
    const viewport = new IntersectionObserver((entries) => {
      visible = entries.at(-1)!.isIntersecting
    })
    const observer = new ResizeObserver(() => instance?.resize())
    viewport.observe(element)
    observer.observe(element)
    $effect(() => {
      if (!visible) return
      const next = option()
      if (next === applied) return
      if (!instance) {
        instance = init(element)
        listen(instance)
      }
      instance.setOption(next)
      applied = next
    })
    return () => {
      viewport.disconnect()
      observer.disconnect()
      instance?.dispose()
    }
  }
}
