<script lang="ts">
  import { getApp, getFilters } from '$lib/state/app.svelte'

  const app = getApp()
  const filters = getFilters()
</script>

<div class="flex flex-col gap-2">
  <div class="text-xs tracking-wider text-muted-foreground uppercase">Morphology</div>
  <div class="flex flex-col gap-0.5">
    {#each filters.classes as { index, label, count } (index)}
      <button
        type="button"
        disabled={count === 0}
        onclick={() => filters.morphologies.toggle(index)}
        class="flex w-full items-center gap-2 rounded px-1.5 py-1 text-left text-sm transition hover:bg-muted/40 disabled:opacity-30"
        class:opacity-40={filters.active && !filters.morphologies.has(index)}
        class:font-medium={filters.morphologies.has(index)}
      >
        <span class="size-3 flex-none rounded-sm" style:background={app.swatches?.[index]}></span>
        <span class="truncate">{label}</span>
        <span class="ml-auto text-xs text-muted-foreground tabular-nums">{count}</span>
      </button>
    {/each}
  </div>
</div>
