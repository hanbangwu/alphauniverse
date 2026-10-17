<script lang="ts">
  import { SHADES } from '$lib/color'
  import { COLOURINGS, type Property, UNITS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import type { Coloured } from '$lib/state/mosaic.svelte'

  interface Props {
    property: Property
    coloured: Coloured
    faded: boolean
  }

  let { property, coloured, faded }: Props = $props()

  const app = getApp()

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
  aria-label="{COLOURINGS[property].label} colour scale"
>
  <figcaption class="flex justify-between font-medium">
    <span>{COLOURINGS[property].label}</span>
    <span class="text-muted-foreground">{UNITS[property]}</span>
  </figcaption>
  <div class="flex justify-between font-mono tabular-nums">
    <span>{format(coloured.low)}</span>
    <span>{format(coloured.high)}</span>
  </div>
  {#if faded}
    {@render bar(SHADES, `measured: ${coloured.measured}`)}
    {#if app.shades}
      {@render bar(app.shades.slice(SHADES.length, -1), `AION: ${coloured.predicted}`)}
    {/if}
  {:else}
    {@render bar(
      SHADES,
      coloured.predicted ? `AION: ${coloured.predicted}` : `measured: ${coloured.measured}`
    )}
  {/if}
  {#if coloured.missing}
    <span class="flex items-center gap-1.5 text-muted-foreground">
      <span class="size-2 rounded-sm" style:background={app.shades?.at(-1)}></span>
      no value: {coloured.missing}
    </span>
  {/if}
</figure>
