<script lang="ts">
  import { SELECTED } from '$lib/color'

  interface Props {
    grid: number
    size: number
    selected: number[]
    hovered: number | null
  }

  let { grid, size, selected, hovered }: Props = $props()

  let canvas = $state<HTMLCanvasElement | null>(null)

  const ratio = $derived(size > 0 ? (globalThis.devicePixelRatio ?? 1) : 1)

  $effect(() => {
    if (!canvas || size <= 0) return
    canvas.width = size * ratio
    canvas.height = size * ratio
  })

  $effect(() => {
    const context = canvas?.getContext('2d')
    if (!context || size <= 0) return

    context.setTransform(ratio, 0, 0, ratio, 0, 0)
    context.clearRect(0, 0, size, size)

    const cell = size / grid
    const chosen = new Set(selected)

    const outline = (index: number, stroke: string, width: number) => {
      context.strokeStyle = stroke
      context.lineWidth = width
      context.strokeRect(
        (index % grid) * cell + width / 2,
        Math.floor(index / grid) * cell + width / 2,
        cell - width,
        cell - width
      )
    }

    for (const index of chosen) outline(index, SELECTED, 2)
    if (hovered !== null && !chosen.has(hovered)) outline(hovered, 'oklch(1 0 0)', 1)
  })
</script>

<canvas bind:this={canvas} class="pointer-events-none absolute inset-0 size-full"></canvas>
