<script lang="ts">
  import PanelFrame from '$lib/components/common/panel-frame.svelte'
  import GalaxyThumb from '$lib/components/galaxy-thumb.svelte'
  import { MORPHOLOGIES, UNLABELLED } from '$lib/labels'
  import type { DataPoint } from 'embedding-atlas'

  interface Props {
    tooltip: DataPoint
    label: string | null
  }

  let { tooltip, label }: Props = $props()

  const galaxy = $derived(Number(tooltip.identifier))
</script>

<div
  class="bg-popover text-popover-foreground flex w-40 flex-col gap-1.5 rounded-md border p-2 shadow-md"
>
  <PanelFrame>
    <GalaxyThumb {galaxy} />
  </PanelFrame>
  <span class="text-sm font-medium">#{galaxy}</span>
  {#if label && tooltip.fields}
    {@const { value, width, predicted } = tooltip.fields}
    <span class="text-xs">
      {label}:
      {value === null
        ? 'no value'
        : `${value.toPrecision(3)}${predicted ? ` ± ${width.toPrecision(2)} (AION)` : ''}`}
    </span>
  {:else}
    <span class="text-muted-foreground text-xs">{MORPHOLOGIES[tooltip.category ?? UNLABELLED]}</span
    >
  {/if}
</div>
