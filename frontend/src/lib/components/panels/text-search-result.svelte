<script lang="ts">
  import ImageTokenFrame from '$lib/components/common/image-token-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { morphologyQuery } from '$lib/data/queries'
  import { MORPHOLOGIES } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    score: number
  }

  let { galaxy, score }: Props = $props()

  const app = getApp()
  const { view, mosaic } = app

  const points = createQuery(() => app.meanPoints)
  const morphology = createQuery(() => morphologyQuery(mosaic, points.data ?? null, galaxy))
</script>

<button
  type="button"
  onclick={() => view.select(galaxy)}
  aria-pressed={view.galaxy.value === galaxy}
  class="flex w-full items-center gap-4 rounded-lg p-2 text-left hover:bg-muted aria-pressed:bg-muted"
>
  <ImageTokenFrame class="size-20 shrink-0">
    <GalaxyThumb {galaxy} />
  </ImageTokenFrame>
  <div class="flex flex-col gap-1">
    <span class="text-sm font-medium">#{galaxy}</span>
    <span class="text-xs text-muted-foreground tabular-nums">
      cos sim {score.toFixed(3)} &bull; {(morphology.data == null
        ? null
        : MORPHOLOGIES[morphology.data]) ?? '-'}
    </span>
  </div>
</button>
