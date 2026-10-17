<script lang="ts">
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { getView } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    galaxy: number
    score?: number
    onselect?: () => void
  }

  let { galaxy, score, onselect }: Props = $props()

  const similarity = getSimilarity()
  const view = getView()
</script>

{#snippet thumb()}
  <PanelFrame>
    <GalaxyThumb {galaxy} predicted={view.predictions.value} />
  </PanelFrame>
{/snippet}

<div class="flex flex-1 flex-col gap-2">
  <span class="text-sm font-medium">
    #{galaxy}{score === undefined ? '' : `: ${similarity.score(score)}`}
  </span>
  {#if onselect}
    <button
      type="button"
      onclick={onselect}
      aria-label="Search from galaxy {galaxy}"
      class="cursor-pointer rounded-lg ring-foreground outline-none hover:ring-2 focus-visible:ring-2"
    >
      {@render thumb()}
    </button>
  {:else}
    {@render thumb()}
  {/if}
</div>
