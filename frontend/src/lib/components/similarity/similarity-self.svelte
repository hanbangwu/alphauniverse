<script lang="ts">
  import Colorbar from '$lib/components/colorbar.svelte'
  import { SIMILARITY } from '$lib/labels'
  import { getView } from '$lib/state/app.svelte'
  import GalaxyTile from './galaxy-tile.svelte'
  import PatchMask from './patch-mask.svelte'
  import PatchPanel from './patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
  const view = getView()
</script>

<div class="flex flex-col gap-5 md:flex-row">
  <GalaxyTile galaxy={similarity.galaxy} />

  {#if similarity.self}
    <PatchMask values={similarity.self} />
  {/if}

  <PatchPanel
    label={similarity.self ? SIMILARITY.label : 'Tokens'}
    describe="Click a patch to query it"
    values={similarity.cells}
    grid={similarity.grid}
    color={similarity.palette}
    title={similarity.caption}
    selected={view.patches.value}
    onselect={(indices) => (view.patches.value = indices)}
    busy={similarity.cells === null || (view.patches.value.length > 0 && similarity.fetching)}
  >
    {#snippet footer()}
      {#if similarity.domain}
        <Colorbar domain={similarity.domain} label={SIMILARITY.label} />
      {/if}
    {/snippet}
  </PatchPanel>
</div>
