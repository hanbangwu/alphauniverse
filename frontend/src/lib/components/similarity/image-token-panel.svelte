<script lang="ts">
  import type { ImageSurvey } from '$lib/client'
  import { OUTLINE, type RGB } from '$lib/color'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import ImageTokenGrid from '$lib/components/image-token/image-token-grid.svelte'
  import type { Snippet } from 'svelte'

  interface Props {
    label: string
    values: ArrayLike<number> | null
    grid: number
    color: ((value: number) => RGB) | null
    mask?: ArrayLike<number> | null
    opacity?: (picked: boolean) => number
    galaxy?: number
    survey?: ImageSurvey
    title?: (value: number, index: number) => string
    describe?: string
    selected?: number[]
    ontoggle?: (index: number) => void
    busy?: boolean
    onhover?: (hovered: boolean) => void
    imageOpacity?: number
    children?: Snippet
  }

  let {
    label,
    values,
    grid,
    color,
    mask,
    opacity,
    galaxy,
    survey,
    title,
    describe,
    selected,
    ontoggle,
    busy = false,
    onhover,
    imageOpacity,
    children
  }: Props = $props()

  const outline = $derived.by(() => {
    const chosen = new Set(selected)
    const edges: string[] = []
    for (const index of chosen) {
      const column = index % grid
      const row = Math.floor(index / grid)
      if (!chosen.has(index - grid)) edges.push(`M${column} ${row}h1`)
      if (!chosen.has(index + grid)) edges.push(`M${column} ${row + 1}h1`)
      if (column === 0 || !chosen.has(index - 1)) edges.push(`M${column} ${row}v1`)
      if (column === grid - 1 || !chosen.has(index + 1)) edges.push(`M${column + 1} ${row}v1`)
    }
    return edges.join('')
  })
</script>

<div class="flex flex-1 flex-col gap-2">
  <span class="text-sm font-medium">{label}</span>

  <div
    role="presentation"
    onpointerenter={() => onhover?.(true)}
    onpointerleave={() => onhover?.(false)}
  >
    <PanelFrame {busy}>
      {#if values && color}
        <ImageTokenGrid
          {values}
          {grid}
          {color}
          {mask}
          {opacity}
          {title}
          {selected}
          {ontoggle}
          label={describe ?? label}
          class={ontoggle ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
        />
      {/if}
      {#if galaxy !== undefined}
        <div
          class="pointer-events-none absolute inset-0 mix-blend-screen transition-opacity duration-150 ease-in-out"
          style:opacity={imageOpacity}
        >
          <GalaxyThumb {galaxy} {survey} />
        </div>
      {/if}
      <svg
        viewBox="0 0 {grid} {grid}"
        class="pointer-events-none absolute inset-0 size-full"
        stroke-linecap="square"
        aria-hidden="true"
      >
        {#each OUTLINE as { stroke, width } (stroke)}
          <path d={outline} {stroke} stroke-width={width} vector-effect="non-scaling-stroke" />
        {/each}
      </svg>
    </PanelFrame>
  </div>

  {@render children?.()}
</div>
