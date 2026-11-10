<script lang="ts">
  import Segmented from '$lib/components/common/segmented.svelte'
  import { Button } from '$lib/components/ui/button'
  import * as Drawer from '$lib/components/ui/drawer'
  import { getView } from '$lib/state/app.svelte'
  import ColourControl from './colour-control.svelte'
  import PanelLeftIcon from '@lucide/svelte/icons/panel-left'

  const view = getView()
  let open = $state(true)
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
        Layout
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
      <Segmented field={view.pointSet} />
      <ColourControl />
    </div>
  </Drawer.Content>
</Drawer.Root>
