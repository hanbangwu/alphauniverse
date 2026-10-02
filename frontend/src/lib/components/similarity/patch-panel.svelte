<script lang="ts">
  import type { RGB } from '$lib/color'
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import PatchGrid from '$lib/components/patch/patch-grid.svelte'
  import type { Snippet } from 'svelte'

  interface Props {
    label: string
    values: ArrayLike<number> | null
    grid: number
    color: ((value: number) => RGB) | null
    opacity?: (picked: boolean) => number
    galaxy?: number
    title?: (value: number, index: number) => string
    describe?: string
    selected?: number[]
    onselect?: (indices: number[]) => void
    busy?: boolean
    onhover?: (hovered: boolean) => void
    imageOpacity?: number
    action?: Snippet
    children?: Snippet
  }

  let {
    label,
    values,
    grid,
    color,
    opacity,
    galaxy,
    title,
    describe,
    selected,
    onselect,
    busy = false,
    onhover,
    imageOpacity,
    action,
    children
  }: Props = $props()
</script>

<div class="flex flex-1 flex-col gap-2">
  <div class="flex items-center justify-between gap-2">
    <span class="text-sm font-medium">{label}</span>
    {@render action?.()}
  </div>

  <div
    role="presentation"
    onpointerenter={() => onhover?.(true)}
    onpointerleave={() => onhover?.(false)}
  >
    <PatchFrame {busy}>
      {#if values && color}
        <PatchGrid
          {values}
          {grid}
          {color}
          {opacity}
          {title}
          {selected}
          {onselect}
          label={describe ?? label}
          class={onselect ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
        />
      {/if}
      {#if galaxy !== undefined}
        <div
          class="pointer-events-none absolute inset-0 mix-blend-screen transition-opacity duration-150 ease-in-out"
          style:opacity={imageOpacity}
        >
          <GalaxyThumb {galaxy} />
        </div>
      {/if}
    </PatchFrame>
  </div>

  {@render children?.()}
</div>
