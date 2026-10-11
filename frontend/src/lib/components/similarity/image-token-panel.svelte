<script lang="ts">
  import type { RGB } from '$lib/color'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import ImageTokenGrid from '$lib/components/image-token/image-token-grid.svelte'
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
    ontoggle?: (index: number) => void
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
    ontoggle,
    busy = false,
    onhover,
    imageOpacity,
    action,
    children
  }: Props = $props()
</script>

<div class="flex flex-1 flex-col gap-2">
  <div class="flex flex-wrap items-center justify-between gap-2">
    <span class="text-sm font-medium">{label}</span>
    {@render action?.()}
  </div>

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
          <GalaxyThumb {galaxy} />
        </div>
      {/if}
    </PanelFrame>
  </div>

  {@render children?.()}
</div>
