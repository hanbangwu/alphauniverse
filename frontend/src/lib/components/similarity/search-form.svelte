<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import { Switch } from '$lib/components/ui/switch'
  import * as Tabs from '$lib/components/ui/tabs'
  import { TOKEN_MODES } from '$lib/data/modes'
  import { getMask } from '$lib/state/app.svelte'
  import MatchCount from './match-count.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
  const display = getMask()

  const mapped = $derived(TOKEN_MODES.some((mode) => similarity.layerOf(mode)))
</script>

<form
  class="sticky -top-6 z-10 -mx-6 -my-2 flex flex-wrap items-center gap-x-5 gap-y-2 bg-popover px-6 py-2"
  onsubmit={(event) => {
    event.preventDefault()
    similarity.submit()
  }}
>
  <Tabs.Root bind:value={similarity.layer}>
    <Tabs.List aria-label="Query map">
      <Tabs.Trigger value="galaxy">This galaxy</Tabs.Trigger>
      <Tabs.Trigger value="population">Population</Tabs.Trigger>
    </Tabs.List>
  </Tabs.Root>
  <label class="flex items-center gap-2 text-sm font-medium">
    <Switch
      bind:checked={display.on.value}
      disabled={!display.on.value && !mapped && !similarity.searched}
    />
    Mask
  </label>
  <label class="flex items-center gap-2 text-sm font-medium">
    <Switch bind:checked={display.invert.value} disabled={!display.on.value} />
    Invert
  </label>
  <div class="ml-auto flex flex-wrap items-center gap-x-3 gap-y-2">
    <MatchCount />
    <Button type="submit" size="sm" disabled={similarity.draft === null}>Search</Button>
  </div>
</form>

{#if similarity.saliencyError}
  <p role="alert" class="text-xs leading-relaxed text-destructive">{similarity.saliencyError}</p>
{/if}
