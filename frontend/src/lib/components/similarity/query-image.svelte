<script lang="ts">
  import type { ImageSurvey } from '$lib/client'
  import { tokenAlpha, tokenColors } from '$lib/color'
  import { imageMode } from '$lib/data/modes'
  import { tokensQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import ImageTokenPanel from './image-token-panel.svelte'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    survey: ImageSurvey
    matched: boolean
  }

  let { survey, matched }: Props = $props()

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const mode = $derived(imageMode(survey))
  const tokens = createQuery(() => tokensQuery(matched ? similarity.galaxy : null, survey))
  const palette = $derived(tokens.data ? tokenColors(tokens.data) : null)
  const layer = $derived(similarity.layerOf(mode))
  const error = $derived(tokens.isError ? errorMessage(tokens.error) : similarity.layerError(mode))

  let hovered = $state(false)
</script>

<div class="flex flex-col gap-5">
  <ImageTokenPanel
    label={SURVEYS[survey].label}
    describe={matched
      ? `Click or drag across ${SURVEYS[survey].label} image tokens to query them`
      : undefined}
    values={layer?.values ?? tokens.data ?? null}
    grid={similarity.grid}
    color={layer?.heat ?? palette}
    mask={layer &&
      similarity.maskOf(layer.values, similarity.imageLayerDomain, display.images.query)}
    opacity={layer ? undefined : (picked) => (hovered ? tokenAlpha(picked) : 1)}
    galaxy={similarity.galaxy}
    {survey}
    title={layer ? similarity.layerCaption : similarity.caption}
    selected={view.tokens[mode].value}
    ontoggle={matched ? (index) => view.tokens[mode].toggle(index) : undefined}
    busy={matched && !tokens.data && !tokens.isError}
    onhover={(value) => (hovered = value)}
    imageOpacity={layer || hovered || !matched ? undefined : tokenAlpha(false)}
  >
    {#if error}
      <p role="alert" class="text-xs leading-relaxed text-destructive">{error}</p>
    {/if}
  </ImageTokenPanel>
</div>
