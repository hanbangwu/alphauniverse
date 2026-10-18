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
    TABLE_MODES,
    imageMode,
    spectrumMode
  } from '$lib/data/modes'
  import type { ModeName } from '$lib/data/similarity'
  import { getApp } from '$lib/state/app.svelte'
  import type { Field } from '$lib/state/field.svelte'
  import type { IndexListField } from '$lib/state/fields.svelte'
  import MaskControls from './mask-controls.svelte'
  import QueryImage from './query-image.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'
  import TablePanel from './table-panel.svelte'
  import WholeMode from './whole-mode.svelte'

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const whole = $derived(view.wholeModes.value)

  function count(modes: readonly ModeName[], fields: IndexListField[]): number {
    return whole
      ? similarity.modes.filter((mode) => modes.includes(mode)).length
      : fields.reduce((total, field) => total + field.value.length, 0)
  }

  function clear(modes: readonly ModeName[], fields: IndexListField[]): void {
    if (whole) view.modes.value = view.modes.value.filter((mode) => !modes.includes(mode))
    else fields.forEach((field) => field.reset())
  }
</script>

{#snippet controls(
  noun: string,
  modes: readonly ModeName[],
  fields: IndexListField[],
  anywhere?: Field<boolean>
)}
  {@const selected = count(modes, fields)}
  {#if selected}
    <span class="text-xs text-muted-foreground">{selected} selected</span>
    <Button
      variant="ghost"
      size="xs"
      aria-label="Clear {noun} selection"
      onclick={() => clear(modes, fields)}
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
        IMAGE_MODES,
        IMAGE_MODES.map((mode) => view.tokens[mode]),
        view.imageAnywhere
      )}
    {/snippet}
    <div class="flex flex-col gap-2">
      <span class="text-sm font-medium">#{similarity.galaxy}</span>
      <div class="flex max-w-2xl flex-col gap-5">
        {#each IMAGE_SURVEYS as survey (survey)}
          <Observed galaxy={similarity.galaxy} {survey} kind="image">
            {#snippet children(matched)}
              <WholeMode mode={imageMode(survey)}>
                <QueryImage {survey} {matched} />
              </WholeMode>
            {/snippet}
          </Observed>
        {/each}
      </div>
    </div>
    {#if display.on.value && similarity.imageLayerDomain}
      <MaskControls
        label="Mask threshold"
        domain={similarity.imageLayerDomain}
        control={display.images.query}
      />
    {/if}
  </CollapsibleGroup>

  <CollapsibleGroup title="Spectra" bind:open={view.open.current.spectra}>
    {#snippet actions()}
      {@render controls(
        'spectrum',
        SPECTRUM_MODES,
        SPECTRUM_MODES.map((mode) => view.tokens[mode]),
        view.spectrumAnywhere
      )}
    {/snippet}
    {#each SPECTRUM_SURVEYS as survey (survey)}
      {@const mode = spectrumMode(survey)}
      {@const layer = similarity.layerOf(mode)}
      <Observed galaxy={similarity.galaxy} {survey} kind="spectrum">
        {#snippet children(matched)}
          <WholeMode {mode}>
            <SpectrumPanel
              galaxy={similarity.galaxy}
              {survey}
              {matched}
              map={layer?.values ?? null}
              heat={layer?.heat ?? null}
              mask={layer &&
                similarity.maskOf(
                  layer.values,
                  similarity.spectrumLayerDomain,
                  display.spectra.query
                )}
              caption={similarity.layerCaption}
              selected={similarity.tokensOf(mode)}
              ontoggle={similarity.toggler(mode, matched)}
              error={similarity.layerError(mode)}
            />
          </WholeMode>
        {/snippet}
      </Observed>
    {/each}
    {#if display.on.value && similarity.spectrumLayerDomain}
      <MaskControls
        label="Mask threshold"
        domain={similarity.spectrumLayerDomain}
        control={display.spectra.query}
      />
    {/if}
  </CollapsibleGroup>

  <CollapsibleGroup title="Catalogue" bind:open={view.open.current.catalogue}>
    {#snippet actions()}
      {@render controls('catalogue', Object.values(TABLE_MODES), [view.tableValues])}
    {/snippet}
    <TablePanel
      galaxy={similarity.galaxy}
      selected={view.tableValues.value}
      ontoggle={(index) => view.tableValues.toggle(index)}
    />
    {#if display.on.value && similarity.tableValueDomain}
      <MaskControls
        label="Mask threshold"
        domain={similarity.tableValueDomain}
        control={display.table}
      />
    {/if}
  </CollapsibleGroup>
</div>
