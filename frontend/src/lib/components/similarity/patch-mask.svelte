<script lang="ts">
  import PatchPanel from './patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'
  import type { Snippet } from 'svelte'

  interface Props {
    values: ArrayLike<number>
    action?: Snippet
    children?: Snippet
  }

  let { values, action, children }: Props = $props()

  const similarity = getSimilarity()
  const bits = $derived(similarity.maskOf(values))
</script>

{#if bits}
  <PatchPanel
    label="Image Mask"
    values={bits}
    grid={similarity.grid}
    color={similarity.maskColor}
    title={similarity.maskTitle}
    {action}
    {children}
  />
{/if}
