<script lang="ts">
  import Segmented from '$lib/components/common/segmented.svelte'
  import * as Select from '$lib/components/ui/select'
  import { colouringQuery, pointsQuery } from '$lib/data/queries'
  import { getApp } from '$lib/state/app.svelte'
  import ColourLegend from './colour-legend.svelte'
  import { createQuery } from '@tanstack/svelte-query'
  import { mode } from 'mode-watcher'

  const { view, mosaic } = getApp()

  const shown = createQuery(() => pointsQuery(mosaic, view.pointSet.value))
  const colouring = createQuery(() =>
    colouringQuery(mosaic, shown.data ?? null, view.property.value, view.everywhere)
  )
</script>

<div class="flex flex-col gap-2">
  <div class="text-xs tracking-wider text-muted-foreground uppercase">Colour</div>
  <Select.Root type="single" bind:value={view.property.value}>
    <Select.Trigger size="sm" class="w-full" aria-label="Colour the map by">
      {view.property.label}
    </Select.Trigger>
    <Select.Content>
      {#each view.property.choices as { value, label } (value)}
        <Select.Item {value} {label} />
      {/each}
    </Select.Content>
  </Select.Root>
  {#if view.predictable}
    <Segmented field={view.source} />
  {/if}
  {#if colouring.data && mode.current}
    {#key colouring.data.name}
      <ColourLegend coloured={colouring.data} scheme={mode.current} />
    {/key}
  {/if}
</div>
