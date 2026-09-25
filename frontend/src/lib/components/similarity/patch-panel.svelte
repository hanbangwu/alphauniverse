<script lang="ts">
  import { type RGB, tokenAlpha } from '$lib/color'
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import PatchGrid from '$lib/components/patch/patch-grid.svelte'

  interface Props {
    label: string
    values: ArrayLike<number> | null
    grid: number
    color: ((value: number) => RGB) | null
    galaxy?: number
    title?: (value: number, index: number) => string
    describe?: string
    selected?: number[]
    onselect?: (indices: number[]) => void
    busy?: boolean
  }

  let {
    label,
    values,
    grid,
    color,
    galaxy,
    title,
    describe,
    selected,
    onselect,
    busy = false
  }: Props = $props()
</script>

<div class="flex flex-1 flex-col gap-2">
  <span class="text-sm font-medium">{label}</span>

  <PatchFrame {busy}>
    {#if galaxy !== undefined}
      <GalaxyThumb {galaxy} />
    {/if}
    {#if values && color}
      <PatchGrid
        {values}
        {grid}
        {color}
        opacity={galaxy === undefined ? undefined : tokenAlpha}
        {title}
        {selected}
        {onselect}
        label={describe ?? label}
        class={onselect ? 'absolute inset-0 cursor-pointer' : 'absolute inset-0'}
      />
    {/if}
  </PatchFrame>
</div>
