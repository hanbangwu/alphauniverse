<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import * as Tabs from '$lib/components/ui/tabs'
  import { getApp } from '$lib/state/app.svelte'
  import type { Field } from '$lib/state/field.svelte'
  import MatchCount from './match-count.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
  const { view } = getApp()
</script>

{#snippet placement(label: string, anywhere: Field<boolean>)}
  <div class="flex items-center gap-2">
    <span class="text-sm font-medium">{label}</span>
    <Tabs.Root
      value={anywhere.value ? 'anywhere' : 'aligned'}
      onValueChange={(value) => (anywhere.value = value === 'anywhere')}
    >
      <Tabs.List aria-label={`${label} matching`}>
        <Tabs.Trigger value="aligned">Same place</Tabs.Trigger>
        <Tabs.Trigger value="anywhere">Any position</Tabs.Trigger>
      </Tabs.List>
    </Tabs.Root>
  </div>
{/snippet}

<form
  class="flex flex-wrap items-center gap-x-4 gap-y-2"
  onsubmit={(event) => {
    event.preventDefault()
    similarity.submit()
  }}
>
  <MatchCount />
  {@render placement('Image', view.imageAnywhere)}
  {@render placement('Spectrum', view.spectrumAnywhere)}
  <Button type="submit" size="sm" disabled={similarity.draft === null}>Search</Button>
</form>
