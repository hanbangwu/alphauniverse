<script lang="ts">
  import type { TokenMode } from '$lib/client'
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
  import { SURVEYS } from '$lib/labels'
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

  const whole = $derived(view.searchBy.value === 'whole')

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

{#snippet controls(noun: string, modes: readonly ModeName[], fields: IndexListField[])}
  {#if count(modes, fields)}
    <Button
      variant="ghost"
      size="sm"
      aria-label="Clear {noun} selection"
      onclick={() => clear(modes, fields)}
    >
      Clear
    </Button>
  {/if}
{/snippet}

{#snippet falseColor(field: Field<boolean>)}
  <label class="flex items-center gap-2 text-sm font-medium">
    <Switch bind:checked={field.value} />
    False color
  </label>
{/snippet}

{#snippet anyPosition(noun: string, modes: readonly TokenMode[], anywhere: Field<boolean>)}
  <label class="flex items-center gap-2 text-sm font-medium">
    <Switch
      aria-label="Any position for {noun} patches"
      bind:checked={anywhere.value}
      disabled={!modes.some((mode) => similarity.shown(mode))}
    />
    Any position
  </label>
{/snippet}

<div class={['flex flex-col gap-6 transition-opacity', similarity.layerStale && 'opacity-50']}>
  <CollapsibleGroup title="Images" bind:open={view.open.current.images}>
    {#snippet actions()}
      {@render controls(
        'image',
        IMAGE_MODES,
        IMAGE_MODES.map((mode) => view.patches[mode])
      )}
    {/snippet}
    <div class="flex flex-wrap items-center gap-x-6 gap-y-2">
      {@render anyPosition('image', IMAGE_MODES, view.imageAnywhere)}
      {@render falseColor(similarity.imageFalseColor)}
    </div>
    <div class="grid max-w-2xl gap-5 md:grid-cols-2">
      {#each IMAGE_SURVEYS as survey (survey)}
        <Observed galaxy={similarity.galaxy} {survey} kind="image">
          <section class="flex flex-col gap-2">
            <h3 class="text-xs tracking-wider text-muted-foreground uppercase">
              {SURVEYS[survey]}
            </h3>
            <WholeMode mode={imageMode(survey)}>
              <QueryImage {survey} />
            </WholeMode>
          </section>
        </Observed>
      {/each}
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
        SPECTRUM_MODES.map((mode) => view.patches[mode])
      )}
    {/snippet}
    <div class="flex flex-wrap items-center gap-x-6 gap-y-2">
      {@render anyPosition('spectrum', SPECTRUM_MODES, view.spectrumAnywhere)}
      {@render falseColor(similarity.spectrumFalseColor)}
    </div>
    {#each SPECTRUM_SURVEYS as survey (survey)}
      {@const mode = spectrumMode(survey)}
      {@const layer = similarity.layerOf(mode)}
      <Observed galaxy={similarity.galaxy} {survey} kind="spectrum">
        {#snippet children(matched)}
          <section class="flex flex-col gap-2">
            <h3 class="text-xs tracking-wider text-muted-foreground uppercase">
              {SURVEYS[survey]}
            </h3>
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
                caption={similarity.correlation}
                selected={similarity.patchesOf(mode)}
                ontoggle={similarity.toggler(mode)}
                error={similarity.saliencyError}
              />
            </WholeMode>
          </section>
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

  <CollapsibleGroup title="Catalog" bind:open={view.open.current.catalog}>
    {#snippet actions()}
      {@render controls('catalog', Object.values(TABLE_MODES), [view.tableValues])}
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
