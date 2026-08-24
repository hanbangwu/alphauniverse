<script lang="ts">
  import type { RGB } from '$lib/color'
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import PatchGrid from '$lib/components/patch/patch-grid.svelte'
  import type { Snippet } from 'svelte'

  interface Props {
    label: string
    values: ArrayLike<number> | null
    grid: number
    color: ((value: number) => RGB) | null
    title?: (value: number, index: number) => string
    badge?: string
    describe?: string
    selected?: number[]
    onselect?: (indices: number[]) => void
    busy?: boolean
    footer?: Snippet
  }

  let {
    label,
    values,
    grid,
    color,
    title,
    badge,
    describe,
    selected,
    onselect,
    busy = false,
    footer
  }: Props = $props()
</script>

<div class="flex flex-1 flex-col gap-2">
  <div class="flex items-baseline justify-between">
    <span class="text-sm font-medium">{label}</span>
    {#if badge}
      <span class="font-mono text-xs text-muted-foreground tabular-nums">{badge}</span>
    {/if}
  </div>

  <PatchFrame {busy}>
    {#if values && color}
      <PatchGrid
        {values}
        {grid}
        {color}
        {title}
        {selected}
        {onselect}
        label={describe ?? label}
        class={onselect ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
      />
    {/if}
  </PatchFrame>

  {@render footer?.()}
</div>
