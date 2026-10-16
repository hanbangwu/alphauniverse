<script lang="ts">
  import CollapsibleGroup from '$lib/components/common/collapsible-group.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Switch } from '$lib/components/ui/switch'
  import { IMAGE_SURVEYS, SPECTRUM_SURVEYS, imageMode, spectrumMode } from '$lib/data/modes'
  import { coverageQuery } from '$lib/data/queries'
  import { SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import type { Field } from '$lib/state/field.svelte'
  import type { IndexListField } from '$lib/state/fields.svelte'
  import GalaxyTile from './galaxy-tile.svelte'
  import QueryImage from './query-image.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'
  import TablePanel from './table-panel.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  const similarity = getSimilarity()
  const { view } = getApp()

  const coverage = createQuery(() => coverageQuery(similarity.galaxy))
</script>

{#snippet controls(fields: IndexListField[], anywhere?: Field<boolean>)}
  {@const count = fields.reduce((total, field) => total + field.value.length, 0)}
  {#if count}
    <span class="text-xs text-muted-foreground">{count} selected</span>
    <Button variant="ghost" size="xs" onclick={() => fields.forEach((field) => field.reset())}>
      Clear
    </Button>
  {/if}
  {#if anywhere}
    <label class="ml-auto flex items-center gap-2 text-xs">
      <Switch
        size="sm"
        checked={anywhere.value}
        onCheckedChange={(checked) => (anywhere.value = checked)}
      />
      Any position
    </label>
  {/if}
{/snippet}

{#snippet absent(text: string)}
  <p class="text-xs text-muted-foreground">{text}</p>
{/snippet}

<div class={['flex flex-col gap-6 transition-opacity', similarity.layerStale && 'opacity-50']}>
  <CollapsibleGroup title="Images" bind:open={view.open.current.images}>
    {#snippet actions()}
      {@render controls(
        IMAGE_SURVEYS.map((survey) => view.tokens[imageMode(survey)]),
        view.imageAnywhere
      )}
    {/snippet}
    <div class="grid max-w-5xl gap-5 md:grid-cols-3">
      <GalaxyTile galaxy={similarity.galaxy} />
      {#each IMAGE_SURVEYS as survey (survey)}
        {#if coverage.data?.[survey]}
          <QueryImage {survey} />
        {:else if coverage.data}
          {@render absent(`No ${SURVEYS[survey].label} image`)}
        {/if}
      {/each}
    </div>
  </CollapsibleGroup>

  <CollapsibleGroup title="Spectra" bind:open={view.open.current.spectra}>
    {#snippet actions()}
      {@render controls(
        SPECTRUM_SURVEYS.map((survey) => view.tokens[spectrumMode(survey)]),
        view.spectrumAnywhere
      )}
    {/snippet}
    {#each SPECTRUM_SURVEYS as survey (survey)}
      {@const mode = spectrumMode(survey)}
      {#if coverage.data?.[survey]}
        <SpectrumPanel
          galaxy={similarity.galaxy}
          {survey}
          map={similarity.layers[mode]}
          heat={similarity.layerHeats[mode]}
          caption={similarity.layerCaption}
          selected={view.tokens[mode].value}
          ontoggle={(index) => view.tokens[mode].toggle(index)}
          error={similarity.layerError(mode)}
        />
      {:else if coverage.data}
        {@render absent(`No ${SURVEYS[survey].label} spectrum`)}
      {/if}
    {/each}
  </CollapsibleGroup>

  <CollapsibleGroup title="Catalogue" bind:open={view.open.current.catalogue}>
    {#snippet actions()}
      {@render controls([view.tableValues])}
    {/snippet}
    <TablePanel
      galaxy={similarity.galaxy}
      selected={view.tableValues.value}
      ontoggle={(index) => view.tableValues.toggle(index)}
    />
  </CollapsibleGroup>
</div>
