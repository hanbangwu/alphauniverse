<script lang="ts">
  import type { ShareName } from '$lib/api'
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { SHARES } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'

  interface Props {
    galaxy: number
    score: number
    shares: [ShareName, number][]
  }

  let { galaxy, score, shares }: Props = $props()

  const { view } = getApp()

  const leading = $derived(
    shares
      .filter(([, share]) => share >= 0.1)
      .toSorted(([, first], [, second]) => second - first)
      .map(([name, share]) => `${SHARES[name].label} ${Math.round(100 * share)}%`)
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
      {score.toFixed(2)} nats &bull; {leading.join(', ')}
    </span>
  </div>
</button>
