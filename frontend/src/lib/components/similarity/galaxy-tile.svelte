<script lang="ts">
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    galaxy: number
    score?: number
  }

  let { galaxy, score }: Props = $props()

  const similarity = getSimilarity()
</script>

<div class="flex flex-1 flex-col gap-2">
  <span class="text-sm font-medium">
    Row #{galaxy}{score === undefined ? '' : `: ${similarity.score(score)}`}
  </span>
  <PatchFrame>
    <GalaxyThumb {galaxy} />
  </PatchFrame>
</div>
