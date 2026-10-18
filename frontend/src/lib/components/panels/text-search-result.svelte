<script lang="ts">
  import type { TextMatches } from '$lib/api'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { getApp } from '$lib/state/app.svelte'

  interface Props {
    galaxy: number
    modality: TextMatches['modalities'][number]
    score: number
  }

  let { galaxy, modality, score }: Props = $props()

  const { view } = getApp()
</script>

<button
  type="button"
  onclick={() => view.select(galaxy)}
  aria-pressed={view.galaxy.value === galaxy}
  class="flex w-full items-center gap-4 rounded-lg p-2 text-left hover:bg-muted aria-pressed:bg-muted"
>
  <PanelFrame class="size-20 shrink-0">
    <GalaxyThumb {galaxy} />
  </PanelFrame>
  <div class="flex flex-col gap-1">
    <span class="text-sm font-medium">#{galaxy}</span>
    <span class="text-xs text-muted-foreground tabular-nums">
      {modality} &bull; cos sim {score.toFixed(3)}
    </span>
  </div>
</button>
