<script lang="ts">
  import { NO_VALUE, SHADES, type Scheme, fadedShades } from '$lib/color'
  import { Button } from '$lib/components/ui/button'
  import { Slider } from '$lib/components/ui/slider'
  import { withinQuery } from '$lib/data/queries'
  import { COLOURINGS, UNITS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import type { Coloured } from '$lib/state/mosaic.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    coloured: Coloured
    scheme: Scheme
  }

  let { coloured, scheme }: Props = $props()

  const { filters, mosaic } = getApp()

  const format = (value: number) => value.toPrecision(3)
  const STEPS = 200
  const span = $derived(coloured.most - coloured.least)
  const at = (step: number) => coloured.least + (span * step) / STEPS
  const stepOf = (value: number) => Math.round(((value - coloured.least) / span) * STEPS)
  const shown = $derived(filters.range.value ?? [coloured.least, coloured.most])
  const within = createQuery(() => withinQuery(mosaic, coloured.name, filters.range.value))
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
  <div class="pointer-events-auto flex flex-col gap-1.5 pt-1">
    <Slider
      type="multiple"
      value={shown.map(stepOf)}
      onValueChange={([low, high]) =>
        (filters.range.value = low === 0 && high === STEPS ? null : [at(low), at(high)])}
      max={STEPS}
      disabled={span <= 0}
    />
    <span class="font-mono tabular-nums">{format(shown[0])} to {format(shown[1])}</span>
    {#if filters.range.value}
      <div class="flex items-center justify-between gap-2">
        <span class="text-muted-foreground">{within.data ?? '…'} in range</span>
        <Button variant="ghost" size="xs" onclick={() => filters.range.reset()}>Reset</Button>
      </div>
    {/if}
  </div>
  {#if coloured.missing}
    <span class="flex items-center gap-1.5 text-muted-foreground">
      <span class="size-2 rounded-sm" style:background={NO_VALUE[scheme]}></span>
      no value: {coloured.missing}
    </span>
  {/if}
</figure>
