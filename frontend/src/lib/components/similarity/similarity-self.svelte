<script lang="ts">
  import { tokenAlpha, tokenColors } from '$lib/color'
  import { Switch } from '$lib/components/ui/switch'
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
  <div class="flex flex-col gap-5 md:flex-row md:justify-center md:*:max-w-xs">
    <GalaxyTile galaxy={similarity.galaxy} />

    <ImageTokenPanel
      label="Image Tokens"
      describe="Click an image token to query it"
      values={similarity.imageCosines ?? tokens.data ?? null}
      grid={similarity.grid}
      color={similarity.imageCosines ? similarity.imageCosineHeat : palette}
      opacity={similarity.imageCosines ? undefined : (picked) => (hovered ? tokenAlpha(picked) : 1)}
      galaxy={similarity.galaxy}
      title={similarity.imageCosines ? similarity.cosine : similarity.caption}
      selected={view.imageTokens.value}
      ontoggle={(index) => view.imageTokens.toggle(index)}
      busy={!tokens.data && !tokens.isError}
      onhover={(value) => (hovered = value)}
      imageOpacity={similarity.imageCosines || hovered ? undefined : tokenAlpha(false)}
    >
      {#snippet action()}
        <div class="flex items-center gap-4">
          <label class="flex items-center gap-2 text-sm font-medium">
            <Switch
              checked={display.on.value}
              disabled={!display.on.value && !similarity.imageCosines && !similarity.searched}
              onCheckedChange={(checked) => (display.on.value = checked)}
            />
            Mask
          </label>
          <label class="flex items-center gap-2 text-sm font-medium">
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
      {#if similarity.imageCosinesError}
        <p role="alert" class="text-xs leading-relaxed text-destructive">
          {similarity.imageCosinesError}
        </p>
      {/if}
    </ImageTokenPanel>

    {#if similarity.imageCosines && similarity.imageCosineDomain}
      <ImageTokenMask
        values={similarity.imageCosines}
        domain={similarity.imageCosineDomain}
        control={display.query}
      >
        <MaskControls domain={similarity.imageCosineDomain} control={display.query} />
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
    map={similarity.spectrumCosines}
    heat={similarity.spectrumCosineHeat}
    caption={similarity.cosine}
    selected={view.spectrumTokens.value}
    ontoggle={(index) => view.spectrumTokens.toggle(index)}
  />
  {#if similarity.spectrumCosinesError}
    <p role="alert" class="text-xs leading-relaxed text-destructive">
      {similarity.spectrumCosinesError}
    </p>
  {/if}
</div>
