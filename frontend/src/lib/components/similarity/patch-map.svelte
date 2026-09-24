<script lang="ts">
  import { extent } from '$lib/data/scores'
  import { SIMILARITY } from '$lib/labels'
  import PatchPanel from './patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    values: Float32Array
  }

  let { values }: Props = $props()

  const similarity = getSimilarity()
</script>

<PatchPanel
  label={SIMILARITY.label}
  badge={similarity.score(extent(values)[1])}
  {values}
  grid={similarity.grid}
  color={similarity.imageHeat}
  title={similarity.score}
/>
