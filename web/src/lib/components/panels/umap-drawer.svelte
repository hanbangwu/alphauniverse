<script lang="ts">
  import { SHADES } from '$lib/color'
  import { Button } from '$lib/components/ui/button'
  import * as Drawer from '$lib/components/ui/drawer'
  import * as NativeSelect from '$lib/components/ui/native-select'
  import { Slider } from '$lib/components/ui/slider'
  import { Switch } from '$lib/components/ui/switch'
  import { coloringQuery, pointsQuery } from '$lib/data/queries'
  import { PROPERTIES } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import type { Colored } from '$lib/state/mosaic.svelte'
  import PanelLeftIcon from '@lucide/svelte/icons/panel-left'
  import { createQuery } from '@tanstack/svelte-query'

  const { view, mosaic, filters } = getApp()
  let open = $state(true)

  const shown = createQuery(() => pointsQuery(mosaic, view.pointSet.value))
  const coloring = createQuery(() =>
    coloringQuery(mosaic, shown.data ?? null, view.property.value, view.everywhere)
  )

  const STEPS = 200
  const gradient = `linear-gradient(to right, ${SHADES.join(', ')})`
  const format = (value: number) => value.toPrecision(3)
  const bound = (colored: Colored, step: number) =>
    step === 0
      ? colored.least
      : step === STEPS
        ? colored.most
        : colored.low + ((colored.high - colored.low) * step) / STEPS
  const stepOf = (colored: Colored, value: number) =>
    value <= colored.least
      ? 0
      : value >= colored.most
        ? STEPS
        : Math.min(
            STEPS - 1,
            Math.max(1, Math.round(((value - colored.low) / (colored.high - colored.low)) * STEPS))
          )
  let dragged = $state<number[] | null>(null)
</script>

{#if view.tab.value === 'umap' && !open}
  <Button
    variant="ghost"
    size="icon"
    class="glass absolute top-3 left-3 z-10"
    aria-label="Show UMAP controls"
    onclick={() => (open = true)}
  >
    <PanelLeftIcon />
  </Button>
{/if}

<Drawer.Root
  direction="left"
  modal={false}
  handleOnly
  bind:open={() => view.tab.value === 'umap' && open, (value) => (open = value)}
>
  <Drawer.Content trapFocus={false}>
    <Drawer.Header class="flex-row items-center justify-between p-2">
      <Drawer.Title class="text-xs tracking-wider text-muted-foreground uppercase">
        UMAP
      </Drawer.Title>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label="Hide UMAP controls"
        onclick={() => (open = false)}
      >
        <PanelLeftIcon />
      </Button>
    </Drawer.Header>
    <div class="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-2 pb-2">
      <label class="flex items-center gap-2 text-sm font-medium">
        <Switch
          bind:checked={
            () => view.pointSet.value === 'any_position',
            (checked) => (view.pointSet.value = checked ? 'any_position' : 'same_place')
          }
        />
        Any position
      </label>
      <div class="flex flex-col gap-2">
        <div class="text-xs tracking-wider text-muted-foreground uppercase">Property</div>
        <NativeSelect.Root
          size="sm"
          class="w-full"
          aria-label="Color the map by"
          bind:value={view.property.value}
        >
          {#each view.property.choices as { value, label } (value)}
            <NativeSelect.Option {value}>{label}</NativeSelect.Option>
          {/each}
        </NativeSelect.Root>
        {#if coloring.data}
          {@const colored = coloring.data}
          {@const steps =
            dragged ??
            (filters.range.value ?? [colored.least, colored.most]).map((value) =>
              stepOf(colored, value)
            )}
          <figure
            class="flex flex-col gap-1.5 text-xs"
            aria-label="{PROPERTIES[colored.property]} color scale"
          >
            <Slider
              class="[&_[data-slot=slider-range]]:bg-transparent [&_[data-slot=slider-track]]:h-2 [&_[data-slot=slider-track]]:rounded-sm [&_[data-slot=slider-track]]:bg-(image:--gradient)"
              style="--gradient: {gradient}"
              type="multiple"
              value={steps}
              onValueChange={(value) => (dragged = value)}
              onValueCommit={([low, high]) => {
                filters.range.value =
                  low === 0 && high === STEPS ? null : [bound(colored, low), bound(colored, high)]
                dragged = null
              }}
              max={STEPS}
              disabled={colored.high === colored.low}
            />
            <div class="flex justify-between font-mono tabular-nums">
              {#each steps as step, index (index)}
                <span>{format(bound(colored, step))}</span>
              {/each}
            </div>
          </figure>
        {/if}
      </div>
    </div>
  </Drawer.Content>
</Drawer.Root>
