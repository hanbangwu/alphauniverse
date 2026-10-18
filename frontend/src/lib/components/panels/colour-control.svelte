<script lang="ts">
  import Segmented from '$lib/components/common/segmented.svelte'
  import * as Select from '$lib/components/ui/select'
  import { getView } from '$lib/state/app.svelte'

  const view = getView()
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
</div>
