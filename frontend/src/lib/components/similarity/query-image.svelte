<script lang="ts">
  import type { ImageSurvey } from '$lib/client'
  import { tokenAlpha, tokenColors } from '$lib/color'
  import { imageMode } from '$lib/data/modes'
  import { tokensQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import ImageTokenMask from './image-token-mask.svelte'
  import ImageTokenPanel from './image-token-panel.svelte'
  import MaskControls from './mask-controls.svelte'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    survey: ImageSurvey
  }

  let { survey }: Props = $props()

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const mode = $derived(imageMode(survey))
  const tokens = createQuery(() => tokensQuery(similarity.galaxy, survey))
  const palette = $derived(tokens.data ? tokenColors(tokens.data) : null)
  const layer = $derived(similarity.layers[mode])
  const domain = $derived(similarity.layerDomains[mode])
  const error = $derived(tokens.isError ? errorMessage(tokens.error) : similarity.layerError(mode))

  let hovered = $state(false)
</script>

<div class="flex flex-col gap-5">
  <ImageTokenPanel
    label={SURVEYS[survey].label}
    describe={`Click a ${SURVEYS[survey].label} image token to query it`}
    values={layer ?? tokens.data ?? null}
    grid={similarity.grid}
    color={layer ? similarity.layerHeats[mode] : palette}
    opacity={layer ? undefined : (picked) => (hovered ? tokenAlpha(picked) : 1)}
    galaxy={similarity.galaxy}
    {survey}
    title={layer ? similarity.layerCaption : similarity.caption}
    selected={view.tokens[mode].value}
    ontoggle={(index) => view.tokens[mode].toggle(index)}
    busy={!tokens.data && !tokens.isError}
    onhover={(value) => (hovered = value)}
    imageOpacity={layer || hovered ? undefined : tokenAlpha(false)}
  >
    {#if error}
      <p role="alert" class="text-xs leading-relaxed text-destructive">{error}</p>
    {/if}
  </ImageTokenPanel>

  {#if layer && domain}
    <ImageTokenMask values={layer} {domain} control={display.query}>
      <MaskControls {domain} control={display.query} />
    </ImageTokenMask>
  {/if}
</div>
