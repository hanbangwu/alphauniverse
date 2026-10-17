<script lang="ts">
  import { NO_VALUE, SHADES, type Scheme, fadedShades } from '$lib/color'
  import { COLOURINGS, UNITS } from '$lib/labels'
  import type { Coloured } from '$lib/state/mosaic.svelte'

  interface Props {
    coloured: Coloured
    scheme: Scheme
  }

  let { coloured, scheme }: Props = $props()

  const format = (value: number) => value.toPrecision(3)
</script>

{#snippet bar(shades: string[], label: string)}
  <div class="flex flex-col gap-0.5">
    <div
      class="h-2 rounded-sm"
      style:background="linear-gradient(to right, {shades.join(', ')})"
      aria-hidden="true"
    ></div>
    <span class="text-muted-foreground">{label}</span>
  </div>
{/snippet}

<figure
  class="pointer-events-none absolute bottom-3 left-3 flex w-56 flex-col gap-1.5 rounded-md border bg-popover/90 p-2 text-xs"
  aria-label="{COLOURINGS[coloured.property].label} colour scale"
>
  <figcaption class="flex justify-between font-medium">
    <span>{COLOURINGS[coloured.property].label}</span>
    <span class="text-muted-foreground">{UNITS[coloured.property]}</span>
  </figcaption>
  <div class="flex justify-between font-mono tabular-nums">
    <span>{format(coloured.low)}</span>
    <span>{format(coloured.high)}</span>
  </div>
  {#if coloured.measured}
    {@render bar(SHADES, `measured: ${coloured.measured}`)}
  {/if}
  {#if coloured.predicted}
    {@render bar(coloured.faded ? fadedShades(scheme) : SHADES, `AION: ${coloured.predicted}`)}
  {/if}
  {#if coloured.missing}
    <span class="flex items-center gap-1.5 text-muted-foreground">
      <span class="size-2 rounded-sm" style:background={NO_VALUE[scheme]}></span>
      no value: {coloured.missing}
    </span>
  {/if}
</figure>
