<script lang="ts">
  import { tokenAlpha } from '$lib/color'
  import { getView } from '$lib/state/app.svelte'
  import GalaxyTile from './galaxy-tile.svelte'
  import PatchMask from './patch-mask.svelte'
  import PatchPanel from './patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'

  const similarity = getSimilarity()
  const view = getView()
</script>

<div class="flex flex-col gap-5 md:flex-row">
  <GalaxyTile galaxy={similarity.galaxy} />

  {#if similarity.imageMap}
    <PatchMask values={similarity.imageMap} />
  {/if}

  <PatchPanel
    label={similarity.imageMap ? 'Image' : 'Image Tokens'}
    describe="Click a patch to query it"
    values={similarity.imageMap ?? similarity.tokens}
    grid={similarity.grid}
    color={similarity.imageMap ? similarity.imageHeat : similarity.palette}
    opacity={similarity.imageMap ? undefined : tokenAlpha}
    galaxy={similarity.galaxy}
    title={similarity.imageMap ? similarity.score : similarity.caption}
    selected={view.patches.value}
    onselect={(indices) => (view.patches.value = indices)}
    busy={similarity.tokens === null || (view.querying && similarity.fetching)}
  />

  <SpectrumPanel
    galaxy={similarity.galaxy}
    map={similarity.spectrumMap}
    selected={view.spans.value}
    onselect={(indices) => (view.spans.value = indices)}
  />
</div>
