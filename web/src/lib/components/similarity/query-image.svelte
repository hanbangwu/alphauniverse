<script module lang="ts">
  import type { RGB } from '$lib/color'
  import { IMAGE_GRID } from '$lib/data/modes'

  const PATCHES = Array.from({ length: IMAGE_GRID ** 2 }, (_, index) => index)
  const BLACK = (): RGB => [0, 0, 0]
  const CLEAR = (): number => 0
</script>

<script lang="ts">
  import type { ImageSurvey } from '$lib/client'
  import { falseColors } from '$lib/color'
  import { imageMode } from '$lib/data/modes'
  import { imageFalseColorQuery } from '$lib/data/queries'
  import { SURVEYS } from '$lib/labels'
  import { getApp } from '$lib/state/app.svelte'
  import ImagePatchPanel from './image-patch-panel.svelte'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    survey: ImageSurvey
  }

  let { survey }: Props = $props()

  const similarity = getSimilarity()
  const { view, mask: display } = getApp()

  const mode = $derived(imageMode(survey))
  const falseColor = $derived(similarity.imageFalseColor.value)
  const colors = createQuery(() =>
    imageFalseColorQuery(
      falseColor ? similarity.galaxy : null,
      survey,
      view.dataSource.value === 'prediction'
    )
  )
  const palette = $derived(colors.data ? falseColors(colors.data) : null)
  const layer = $derived(similarity.layerOf(mode))
  const error = $derived(colors.isError ? colors.error.message : similarity.saliencyError)
</script>

<ImagePatchPanel
  label={SURVEYS[survey]}
  values={layer?.values ?? PATCHES}
  color={layer?.heat ?? palette ?? BLACK}
  mask={layer && similarity.maskOf(layer.values, similarity.imageLayerDomain, display.images.query)}
  opacity={layer || palette ? undefined : CLEAR}
  galaxy={similarity.galaxy}
  {survey}
  title={layer ? similarity.correlation : undefined}
  selected={similarity.patchesOf(mode)}
  ontoggle={similarity.toggler(mode)}
  busy={falseColor && !colors.data && !colors.isError}
  imageOpacity={!layer && palette ? 0 : undefined}
>
  {#if error}
    <p role="alert" class="text-xs leading-relaxed text-destructive">{error}</p>
  {/if}
</ImagePatchPanel>
