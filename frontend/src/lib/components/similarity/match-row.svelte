<script lang="ts">
  import GalaxyTile from './galaxy-tile.svelte'
  import PatchMap from './patch-map.svelte'
  import PatchMask from './patch-mask.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    galaxy: number
    index: number
  }

  let { galaxy, index }: Props = $props()

  const similarity = getSimilarity()
  const values = $derived(similarity.rowAt(index))
</script>

<div class="flex flex-col gap-5 pt-6 md:flex-row">
  <GalaxyTile {galaxy} />
  <PatchMask {values} />
  <PatchMap {values} best={similarity.scoreAt(index)} />
</div>
