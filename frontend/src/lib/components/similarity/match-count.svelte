<script lang="ts">
  import { Input } from '$lib/components/ui/input'
  import { getSearch } from '$lib/state/app.svelte'

  const search = getSearch()
  const error = $props.id()

  const invalid = $derived(search.count === null)
</script>

<label
  class="flex items-center gap-2 text-sm font-medium"
  title="How many galaxies the search returns."
>
  Show top
  <Input
    type="number"
    min={search.range.minimum}
    max={search.range.maximum}
    value={search.matches.value}
    oninput={(event) => (search.matches.value = event.currentTarget.value)}
    aria-invalid={invalid}
    aria-describedby={invalid ? error : undefined}
    class="w-20"
  />
</label>
{#if invalid}
  <span id={error} role="alert" class="text-xs text-destructive">
    Enter a whole number from {search.range.minimum} to {search.range.maximum}.
  </span>
{/if}
