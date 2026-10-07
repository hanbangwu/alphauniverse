<script lang="ts">
  import PatchFrame from '$lib/components/common/patch-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { MORPHOLOGIES, UNLABELLED } from '$lib/labels'
  import type { DataPoint } from 'embedding-atlas'

  interface Props {
    tooltip: DataPoint
  }

  let { tooltip }: Props = $props()

  const galaxy = $derived(Number(tooltip.identifier))
</script>

<div
  class="bg-popover text-popover-foreground flex w-40 flex-col gap-1.5 rounded-md border p-2 shadow-md"
>
  <PatchFrame>
    <GalaxyThumb {galaxy} />
  </PatchFrame>
  <span class="text-sm font-medium">#{galaxy}</span>
  <span class="text-muted-foreground text-xs">{MORPHOLOGIES[tooltip.category ?? UNLABELLED]}</span>
</div>
