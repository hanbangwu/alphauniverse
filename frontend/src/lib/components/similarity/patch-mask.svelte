<script lang="ts">
  import PatchPanel from './patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    values: ArrayLike<number>
  }

  let { values }: Props = $props()

  const similarity = getSimilarity()
  const bits = $derived(similarity.maskOf(values))
</script>

{#if bits}
  <PatchPanel
    label="Mask"
    values={bits}
    grid={similarity.grid}
    color={similarity.maskColor}
    title={similarity.maskTitle}
  />
{/if}
