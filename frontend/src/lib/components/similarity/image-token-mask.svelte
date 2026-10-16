<script lang="ts">
  import type { Extent } from '$lib/data/similarity'
  import type { Threshold } from '$lib/state/mask.svelte'
  import ImageTokenPanel from './image-token-panel.svelte'
  import { getSimilarity } from './similarity.svelte'
  import type { Snippet } from 'svelte'

  interface Props {
    values: ArrayLike<number>
    domain: Extent | null
    control: Threshold
    children?: Snippet
  }

  let { values, domain, control, children }: Props = $props()

  const similarity = getSimilarity()
  const bits = $derived(similarity.maskOf(values, domain, control))
</script>

{#if bits}
  <ImageTokenPanel
    label="Image Mask"
    values={bits}
    grid={similarity.grid}
    color={similarity.maskColor}
    title={similarity.maskTitle}
    {children}
  />
{/if}
