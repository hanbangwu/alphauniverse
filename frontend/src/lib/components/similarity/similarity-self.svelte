<script lang="ts">
  import { tokenAlpha, tokenColors } from '$lib/color'
  import { Switch } from '$lib/components/ui/switch'
  import { tokensQuery } from '$lib/data/queries'
  import { getApp } from '$lib/state/app.svelte'
  import GalaxyTile from './galaxy-tile.svelte'
  import MaskControls from './mask-controls.svelte'
  import PatchMask from './patch-mask.svelte'
  import PatchPanel from './patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const tokens = createQuery(() => tokensQuery(similarity.galaxy))
  const palette = $derived(tokens.data ? tokenColors(tokens.data) : null)

  let hovered = $state(false)
</script>

<div class="flex flex-col gap-5 md:flex-row">
  <GalaxyTile galaxy={similarity.galaxy} />

  <PatchPanel
    label="Image Tokens"
    describe="Click a patch to query it"
    values={similarity.imageMap ?? tokens.data ?? null}
    grid={similarity.grid}
    color={similarity.imageMap ? similarity.imageHeat : palette}
    opacity={similarity.imageMap ? undefined : (picked) => (hovered ? tokenAlpha(picked) : 1)}
    galaxy={similarity.galaxy}
    title={similarity.imageMap ? similarity.score : similarity.caption}
    selected={view.patches.value}
    ontoggle={(index) => view.patches.toggle(index)}
    busy={!tokens.data}
    onhover={(value) => (hovered = value)}
    imageOpacity={similarity.imageMap || hovered ? undefined : tokenAlpha(false)}
  >
    {#snippet action()}
      <label class="flex items-center gap-2 text-sm font-medium">
        <Switch
          checked={display.on.value}
          disabled={!similarity.searched}
          onCheckedChange={(checked) => (display.on.value = checked)}
        />
        Mask
      </label>
    {/snippet}
  </PatchPanel>

  {#if similarity.imageMap && similarity.imageDomain}
    <PatchMask values={similarity.imageMap}>
      {#snippet action()}
        <label class="flex items-center gap-2 text-sm font-medium">
          <Switch
            checked={display.invert.value}
            onCheckedChange={(checked) => (display.invert.value = checked)}
          />
          Invert
        </label>
      {/snippet}
      <MaskControls domain={similarity.imageDomain} />
    </PatchMask>
  {/if}

  <SpectrumPanel
    galaxy={similarity.galaxy}
    map={similarity.spectrumMap}
    selected={view.spans.value}
    ontoggle={(index) => view.spans.toggle(index)}
  />
</div>
