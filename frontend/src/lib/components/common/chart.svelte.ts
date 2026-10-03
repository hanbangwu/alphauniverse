import type { TooltipComponentOption } from 'echarts/components'

export const TOOLTIP: TooltipComponentOption = {
  confine: true,
  backgroundColor: 'rgba(0, 0, 0, 0.75)',
  borderWidth: 0,
  padding: [2, 6],
  textStyle: { color: '#fff', fontFamily: 'monospace', fontSize: 10 }
}
