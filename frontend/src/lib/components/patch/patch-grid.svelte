<script lang="ts">
  import type { RGB } from '$lib/color'
  import { PatchCursor } from './patch-cursor.svelte'
  import PatchHeat from './patch-heat.svelte'
  import PatchOverlay from './patch-overlay.svelte'
  import { IsInViewport } from 'runed'

  interface Props {
    values: ArrayLike<number>
    grid: number
    color: (value: number) => RGB
    title?: (value: number, index: number) => string
    selected?: number[]
    onselect?: (indices: number[]) => void
    class?: string
    label: string
  }

  let {
    values,
    grid,
    color,
    title,
    selected = [],
    onselect,
    class: className,
    label
  }: Props = $props()

  let box = $state<HTMLElement | null>(null)
  let size = $state(0)

  const viewport = new IsInViewport(() => box)

  const cursor = new PatchCursor({
    grid: () => grid,
    size: () => size,
    box: () => box,
    selected: () => selected,
    onselect: (indices) => onselect?.(indices)
  })

  const interactive = $derived(Boolean(onselect))

  const tip = $derived(
    cursor.hovered === null || !title ? null : title(values[cursor.hovered], cursor.hovered)
  )
</script>

<svelte:element
  this={interactive ? 'button' : 'div'}
  bind:this={box}
  bind:clientWidth={size}
  type={interactive ? 'button' : undefined}
  class={className}
  role={interactive ? undefined : 'img'}
  aria-label={label}
  onpointermove={cursor.onpointermove}
  onpointerleave={cursor.onpointerleave}
  onclick={cursor.onclick}
>
  <PatchHeat {values} {grid} {color} visible={viewport.current} />
  <PatchOverlay {grid} {size} {selected} hovered={cursor.hovered} />

  {#if tip}
    <output
      aria-live="polite"
      class="pointer-events-none absolute bottom-1 left-1 rounded bg-black/75 px-1.5 py-0.5 font-mono text-[10px] text-white tabular-nums"
    >
      {tip}
    </output>
  {/if}
</svelte:element>
