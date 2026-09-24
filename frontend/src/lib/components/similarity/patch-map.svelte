<script lang="ts">
  import { SIMILARITY } from '$lib/labels'
  import { DECIMALS } from '$lib/state/mask.svelte'
  import PatchPanel from './patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    values: Float32Array
    best: number
  }

  let { values, best }: Props = $props()

  const similarity = getSimilarity()
</script>

<PatchPanel
  label={SIMILARITY.label}
  badge={`${SIMILARITY.short} ${best.toFixed(DECIMALS)}`}
  {values}
  grid={similarity.grid}
  color={similarity.imageHeat}
  title={similarity.score}
/>
