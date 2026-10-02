<script lang="ts">
  import PatchPanel from './patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    values: Float32Array
  }

  let { values }: Props = $props()

  const similarity = getSimilarity()
</script>

<PatchPanel
  label="Image Tokens"
  {values}
  grid={similarity.grid}
  color={similarity.imageHeat}
  title={similarity.score}
/>
