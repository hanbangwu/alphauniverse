<script lang="ts">
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import type { ModeName } from '$lib/data/similarity'
  import { MODES } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'

  interface Props {
    galaxy: number
    score: number
    contributions: [ModeName, number][]
  }

  let { galaxy, score, contributions }: Props = $props()

  const { view } = getApp()

  const lead = $derived(contributions.reduce((total, [, value]) => total + value, 0))
  const leading = $derived(
    contributions
      .filter(([, value]) => lead > 0 && value >= 0.1 * lead)
      .toSorted(([, first], [, second]) => second - first)
      .map(([name, value]) => `${MODES[name]} ${Math.round((100 * value) / lead)}%`)
  )
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
    <span class="text-sm font-medium"><GalaxyName {galaxy} /></span>
    <span class="text-xs text-muted-foreground tabular-nums">
      cosine {score.toFixed(3)} &bull; {leading.join(', ')}
    </span>
  </div>
</button>
