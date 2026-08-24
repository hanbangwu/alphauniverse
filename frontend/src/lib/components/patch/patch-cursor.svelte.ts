interface Options {
  grid: () => number
  size: () => number
  box: () => HTMLElement | null
  selected: () => number[]
  onselect: (indices: number[]) => void
}

export class PatchCursor {
  hovered = $state<number | null>(null)

  constructor(private readonly options: Options) {}

  at(event: { clientX: number; clientY: number }): number | null {
    const box = this.options.box()
    const grid = this.options.grid()
    if (!box || this.options.size() <= 0) return null

    const bounds = box.getBoundingClientRect()
    const cell = bounds.width / grid
    const column = Math.floor((event.clientX - bounds.left) / cell)
    const row = Math.floor((event.clientY - bounds.top) / cell)
    if (column < 0 || row < 0 || column >= grid || row >= grid) return null
    return row * grid + column
  }

  onpointermove = (event: PointerEvent): void => {
    this.hovered = this.at(event)
  }

  onpointerleave = (): void => {
    this.hovered = null
  }

  onclick = (event: MouseEvent): void => {
    if (event.detail === 0) return
    const index = this.at(event)
    if (index === null) return

    const selected = this.options.selected()
    this.options.onselect(
      selected.includes(index) ? selected.filter((value) => value !== index) : [...selected, index]
    )
  }
}
