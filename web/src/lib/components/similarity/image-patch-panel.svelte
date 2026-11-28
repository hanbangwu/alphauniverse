<script lang="ts">
  import type { GalaxyId, ImageSurvey } from '$lib/client'
  import { OUTLINE, type RGB } from '$lib/color'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import ImagePatchGrid from '$lib/components/image-patch/image-patch-grid.svelte'
  import { IMAGE_GRID } from '$lib/data/modes'
  import { getView } from '$lib/state/app.svelte'
  import type { Snippet } from 'svelte'

  interface Props {
    label: string
    values: ArrayLike<number> | null
    color: ((value: number) => RGB) | null
    mask?: ArrayLike<number> | null
    opacity?: (picked: boolean) => number
    galaxy?: GalaxyId
    survey?: ImageSurvey
    title?: (value: number, index: number) => string
    selected?: number[]
    ontoggle?: (index: number) => void
    busy?: boolean
    imageOpacity?: number
    children?: Snippet
  }

  let {
    label,
    values,
    color,
    mask,
    opacity,
    galaxy,
    survey,
    title,
    selected,
    ontoggle,
    busy = false,
    imageOpacity,
    children
  }: Props = $props()

  const view = getView()

  let painted = $state.raw<number[]>([])

  const outline = $derived.by(() => {
    const chosen = new Set(selected).symmetricDifference(new Set(painted))
    const edges: string[] = []
    for (const index of chosen) {
      const column = index % IMAGE_GRID
      const row = Math.floor(index / IMAGE_GRID)
      if (!chosen.has(index - IMAGE_GRID)) edges.push(`M${column} ${row}h1`)
      if (!chosen.has(index + IMAGE_GRID)) edges.push(`M${column} ${row + 1}h1`)
      if (column === 0 || !chosen.has(index - 1)) edges.push(`M${column} ${row}v1`)
      if (column === IMAGE_GRID - 1 || !chosen.has(index + 1)) edges.push(`M${column + 1} ${row}v1`)
    }
    return edges.join('')
  })
</script>

<div class="flex flex-1 flex-col gap-2">
  <PanelFrame {busy}>
    {#if values && color}
      <ImagePatchGrid
        {values}
        {color}
        {mask}
        {opacity}
        {title}
        {selected}
        bind:painted
        {ontoggle}
        label={ontoggle ? `Click or drag across ${label} image patches to query them` : label}
        class={ontoggle ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
      />
    {/if}
    {#if galaxy !== undefined}
      <div
        class="pointer-events-none absolute inset-0 mix-blend-screen"
        style:opacity={imageOpacity}
      >
        <GalaxyThumb {galaxy} {survey} predicted={view.dataSource.value === 'prediction'} />
      </div>
    {/if}
    <svg
      viewBox="0 0 {IMAGE_GRID} {IMAGE_GRID}"
      class="pointer-events-none absolute inset-0 size-full"
      stroke-linecap="square"
      aria-hidden="true"
    >
      {#each OUTLINE as { stroke, width } (stroke)}
        <path d={outline} {stroke} stroke-width={width} vector-effect="non-scaling-stroke" />
      {/each}
    </svg>
  </PanelFrame>

  {@render children?.()}
</div>
