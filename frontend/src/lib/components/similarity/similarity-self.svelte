<script lang="ts">
  import CollapsibleGroup from '$lib/components/common/collapsible-group.svelte'
  import Observed from '$lib/components/common/observed.svelte'
  import { Button } from '$lib/components/ui/button'
  import { Switch } from '$lib/components/ui/switch'
  import {
    IMAGE_MODES,
    IMAGE_SURVEYS,
    SPECTRUM_MODES,
    SPECTRUM_SURVEYS,
    spectrumMode
  } from '$lib/data/modes'
  import { getApp } from '$lib/state/app.svelte'
  import type { Field } from '$lib/state/field.svelte'
  import type { IndexListField } from '$lib/state/fields.svelte'
  import GalaxyTile from './galaxy-tile.svelte'
  import MaskControls from './mask-controls.svelte'
  import QueryImage from './query-image.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'
  import TablePanel from './table-panel.svelte'

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()
</script>

{#snippet controls(noun: string, fields: IndexListField[], anywhere?: Field<boolean>)}
  {@const count = fields.reduce((total, field) => total + field.value.length, 0)}
  {#if count}
    <span class="text-xs text-muted-foreground">{count} selected</span>
    <Button
      variant="ghost"
      size="xs"
      aria-label="Clear {noun} selection"
      onclick={() => fields.forEach((field) => field.reset())}
    >
      Clear
    </Button>
  {/if}
  {#if anywhere}
    <label class="ml-auto flex items-center gap-2 text-xs">
      <Switch size="sm" aria-label="Any position for {noun} tokens" bind:checked={anywhere.value} />
      Any position
    </label>
  {/if}
{/snippet}

<div class={['flex flex-col gap-6 transition-opacity', similarity.layerStale && 'opacity-50']}>
  <CollapsibleGroup title="Images" bind:open={view.open.current.images}>
    {#snippet actions()}
      {@render controls(
        'image',
        IMAGE_MODES.map((mode) => view.tokens[mode]),
        view.imageAnywhere
      )}
    {/snippet}
    <div class="grid max-w-5xl gap-5 md:grid-cols-3">
      <GalaxyTile galaxy={similarity.galaxy} />
      {#each IMAGE_SURVEYS as survey (survey)}
        <Observed galaxy={similarity.galaxy} {survey} kind="image">
          <QueryImage {survey} />
        </Observed>
      {/each}
    </div>
    {#if display.on.value && similarity.imageLayerDomain}
      <div class="flex flex-col gap-1 md:max-w-xs">
        <span class="text-xs text-muted-foreground">Query mask threshold</span>
        <MaskControls domain={similarity.imageLayerDomain} control={display.query} />
      </div>
    {/if}
  </CollapsibleGroup>

  <CollapsibleGroup title="Spectra" bind:open={view.open.current.spectra}>
    {#snippet actions()}
      {@render controls(
        'spectrum',
        SPECTRUM_MODES.map((mode) => view.tokens[mode]),
        view.spectrumAnywhere
      )}
    {/snippet}
    {#each SPECTRUM_SURVEYS as survey (survey)}
      {@const mode = spectrumMode(survey)}
      <Observed galaxy={similarity.galaxy} {survey} kind="spectrum">
        <SpectrumPanel
          galaxy={similarity.galaxy}
          {survey}
          map={similarity.layerOf(mode)?.values ?? null}
          heat={similarity.layerOf(mode)?.heat ?? null}
          caption={similarity.layerCaption}
          selected={view.tokens[mode].value}
          ontoggle={(index) => view.tokens[mode].toggle(index)}
          error={similarity.layerError(mode)}
        />
      </Observed>
    {/each}
  </CollapsibleGroup>

  <CollapsibleGroup title="Catalogue" bind:open={view.open.current.catalogue}>
    {#snippet actions()}
      {@render controls('catalogue', [view.tableValues])}
    {/snippet}
    <TablePanel
      galaxy={similarity.galaxy}
      selected={view.tableValues.value}
      ontoggle={(index) => view.tableValues.toggle(index)}
    />
  </CollapsibleGroup>
</div>
