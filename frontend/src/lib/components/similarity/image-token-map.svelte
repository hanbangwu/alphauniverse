<script lang="ts">
  import ImageTokenPanel from './image-token-panel.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    values: Float32Array
    selected: number[]
  }

  let { values, selected }: Props = $props()

  const similarity = getSimilarity()
</script>

<ImageTokenPanel
  label="Image Tokens"
  {values}
  grid={similarity.grid}
  color={similarity.imageHeat}
  title={similarity.score}
  {selected}
/>
