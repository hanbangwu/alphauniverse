<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import { Input } from '$lib/components/ui/input'
  import * as NativeSelect from '$lib/components/ui/native-select'
  import { Switch } from '$lib/components/ui/switch'
  import { PATCH_MODES } from '$lib/data/modes'
  import { getMask, getSearch, getView } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
  const display = getMask()
  const view = getView()
  const search = getSearch()
  const error = $props.id()

  const mapped = $derived(PATCH_MODES.some((mode) => similarity.layerOf(mode)))
  const invalid = $derived(search.count === null)
</script>

<form
  class="flex flex-wrap items-center gap-x-5 gap-y-2"
  onsubmit={(event) => {
    event.preventDefault()
    similarity.submit()
  }}
>
  <label class="flex items-center gap-2 text-sm font-medium">
    Search by
    <NativeSelect.Root size="sm" bind:value={view.searchBy.value}>
      <NativeSelect.Option value="patch">Patch</NativeSelect.Option>
      <NativeSelect.Option value="whole">Whole mode</NativeSelect.Option>
    </NativeSelect.Root>
  </label>
  {#if view.searchBy.value === 'patch'}
    <label class="flex items-center gap-2 text-sm font-medium">
      Highlight by
      <NativeSelect.Root size="sm" bind:value={similarity.layer.value}>
        <NativeSelect.Option value="self">Self</NativeSelect.Option>
        <NativeSelect.Option value="all">All</NativeSelect.Option>
      </NativeSelect.Root>
    </label>
  {/if}
  <label class="flex items-center gap-2 text-sm font-medium">
    <Switch
      bind:checked={display.on.value}
      disabled={!display.on.value && !mapped && !similarity.searched}
    />
    Mask
  </label>
  {#if display.on.value}
    <label class="flex items-center gap-2 text-sm font-medium">
      <Switch bind:checked={display.invert.value} />
      Invert
    </label>
  {/if}
  <div class="ml-auto flex flex-wrap items-center gap-x-3 gap-y-2">
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
    <Button type="submit" size="sm" disabled={similarity.draft === null}>Search</Button>
  </div>
</form>

{#if similarity.saliencyError}
  <p role="alert" class="text-xs leading-relaxed text-destructive">{similarity.saliencyError}</p>
{/if}
