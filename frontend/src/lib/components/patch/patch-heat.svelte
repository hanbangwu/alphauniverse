<script lang="ts">
  import type { RGB } from '$lib/color'

  interface Props {
    values: ArrayLike<number>
    grid: number
    color: (value: number) => RGB
    visible?: boolean
  }

  let { values, grid, color, visible = true }: Props = $props()

  let canvas = $state<HTMLCanvasElement | null>(null)

  $effect(() => {
    const context = visible ? canvas?.getContext('2d') : null
    if (!context) return

    const image = context.createImageData(grid, grid)
    for (let i = 0; i < grid * grid; i++) {
      const [r, g, b] = color(values[i])
      image.data[i * 4] = r
      image.data[i * 4 + 1] = g
      image.data[i * 4 + 2] = b
      image.data[i * 4 + 3] = 255
    }
    context.putImageData(image, 0, 0)
  })
</script>

<canvas
  bind:this={canvas}
  width={grid}
  height={grid}
  class="absolute inset-0 size-full"
  style="image-rendering: pixelated"
></canvas>
