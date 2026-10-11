<script lang="ts">
  import { tokenAlpha, tokenColors } from '$lib/color'
  import { Switch } from '$lib/components/ui/switch'
  import * as Tabs from '$lib/components/ui/tabs'
  import { tokensQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { getApp } from '$lib/state/app.svelte'
  import GalaxyTile from './galaxy-tile.svelte'
  import ImageTokenMask from './image-token-mask.svelte'
  import ImageTokenPanel from './image-token-panel.svelte'
  import MaskControls from './mask-controls.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'
  import TablePanel from './table-panel.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const tokens = createQuery(() => tokensQuery(similarity.galaxy))
  const palette = $derived(tokens.data ? tokenColors(tokens.data) : null)

  let hovered = $state(false)
</script>

<div class="flex flex-col gap-5">
  <Tabs.Root bind:value={similarity.layer} class="self-center">
    <Tabs.List aria-label="Query map">
      <Tabs.Trigger value="galaxy">This galaxy</Tabs.Trigger>
      <Tabs.Trigger value="population">Population</Tabs.Trigger>
    </Tabs.List>
  </Tabs.Root>

  <div class="flex flex-col gap-5 md:flex-row md:justify-center md:*:max-w-xs">
    <GalaxyTile galaxy={similarity.galaxy} />

    <ImageTokenPanel
      label="Image Tokens"
      describe="Click an image token to query it"
      values={similarity.imageLayer ?? tokens.data ?? null}
      grid={similarity.grid}
      color={similarity.imageLayer ? similarity.imageLayerHeat : palette}
      opacity={similarity.imageLayer ? undefined : (picked) => (hovered ? tokenAlpha(picked) : 1)}
      galaxy={similarity.galaxy}
      title={similarity.imageLayer ? similarity.layerCaption : similarity.caption}
      selected={view.imageTokens.value}
      ontoggle={(index) => view.imageTokens.toggle(index)}
      busy={!tokens.data && !tokens.isError}
      onhover={(value) => (hovered = value)}
      imageOpacity={similarity.imageLayer || hovered ? undefined : tokenAlpha(false)}
    >
      {#snippet action()}
        <div class="flex flex-wrap items-center gap-x-4 gap-y-1">
          <label class="flex items-center gap-2 text-sm font-medium whitespace-nowrap">
            <Switch
              checked={display.on.value}
              disabled={!display.on.value && !similarity.imageLayer && !similarity.searched}
              onCheckedChange={(checked) => (display.on.value = checked)}
            />
            Mask
          </label>
          <label class="flex items-center gap-2 text-sm font-medium whitespace-nowrap">
            <Switch
              checked={view.imageAnywhere.value}
              onCheckedChange={(checked) => (view.imageAnywhere.value = checked)}
            />
            Any position
          </label>
          <label class="flex items-center gap-2 text-sm font-medium whitespace-nowrap">
            <Switch
              checked={display.invert.value}
              disabled={!display.on.value}
              onCheckedChange={(checked) => (display.invert.value = checked)}
            />
            Invert
          </label>
        </div>
      {/snippet}
      {#if tokens.isError}
        <p role="alert" class="text-xs leading-relaxed text-destructive">
          {errorMessage(tokens.error)}
        </p>
      {/if}
      {#if similarity.imageLayerError}
        <p role="alert" class="text-xs leading-relaxed text-destructive">
          {similarity.imageLayerError}
        </p>
      {/if}
    </ImageTokenPanel>

    {#if similarity.imageLayer && similarity.imageLayerDomain}
      <ImageTokenMask
        values={similarity.imageLayer}
        domain={similarity.imageLayerDomain}
        control={display.query}
      >
        <MaskControls domain={similarity.imageLayerDomain} control={display.query} />
      </ImageTokenMask>
    {/if}

    <TablePanel
      galaxy={similarity.galaxy}
      selected={view.tableValues.value}
      ontoggle={(index) => view.tableValues.toggle(index)}
    />
  </div>

  <SpectrumPanel
    galaxy={similarity.galaxy}
    map={similarity.spectrumLayer}
    heat={similarity.spectrumLayerHeat}
    caption={similarity.layerCaption}
    selected={view.spectrumTokens.value}
    ontoggle={(index) => view.spectrumTokens.toggle(index)}
  >
    {#snippet action()}
      <label class="flex items-center gap-2 text-sm font-medium whitespace-nowrap">
        <Switch
          checked={view.spectrumAnywhere.value}
          onCheckedChange={(checked) => (view.spectrumAnywhere.value = checked)}
        />
        Any position
      </label>
    {/snippet}
  </SpectrumPanel>
  {#if similarity.spectrumLayerError}
    <p role="alert" class="text-xs leading-relaxed text-destructive">
      {similarity.spectrumLayerError}
    </p>
  {/if}
</div>
