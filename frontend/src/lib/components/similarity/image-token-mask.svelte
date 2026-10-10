<script lang="ts">
  import ImageTokenPanel from './image-token-panel.svelte'
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
  <ImageTokenPanel
    label="Image Mask"
    values={bits}
    grid={similarity.grid}
    color={similarity.maskColor}
    title={similarity.maskTitle}
    {action}
    {children}
  />
{/if}
