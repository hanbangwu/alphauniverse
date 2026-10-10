<script lang="ts">
  import { OBSERVATIONS } from '$lib/labels'
  import GalaxyTile from './galaxy-tile.svelte'
  import ImageTokenMap from './image-token-map.svelte'
  import ImageTokenMask from './image-token-mask.svelte'
  import { getSimilarity } from './similarity.svelte'
  import SpectrumPanel from './spectrum-panel.svelte'

  interface Props {
    galaxy: number
    index: number
  }

  let { galaxy, index }: Props = $props()

  const similarity = getSimilarity()
  const values = $derived(similarity.imageMapAt(index))
  const predicted = $derived(similarity.predictedAt(index))
  const elsewhere = $derived(
    predicted.filter((column) => column !== 'desi').map((column) => OBSERVATIONS[column])
  )
</script>

<div class="flex flex-col gap-5 pt-6">
  <div class="flex flex-col gap-5 md:flex-row md:justify-center md:*:max-w-xs">
    <GalaxyTile {galaxy} score={similarity.scoreAt(index)} />
    <ImageTokenMap {values} />
    <ImageTokenMask {values} />
  </div>
  {#if elsewhere.length}
    <p class="text-xs text-muted-foreground">
      No {elsewhere.join(', ')}: matched on AION's prediction
    </p>
  {/if}
  <SpectrumPanel
    {galaxy}
    map={similarity.spectrumMapAt(index)}
    absent={predicted.includes('desi')
      ? "No DESI spectrum: matched on AION's prediction"
      : undefined}
  />
</div>
