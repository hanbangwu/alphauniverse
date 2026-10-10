<script lang="ts">
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
</script>

<div class="flex flex-col gap-5 pt-6">
  <div class="flex flex-col gap-5 md:flex-row md:justify-center md:*:max-w-xs">
    <GalaxyTile {galaxy} score={similarity.scoreAt(index)} />
    <ImageTokenMap {values} />
    <ImageTokenMask {values} />
  </div>
  <SpectrumPanel {galaxy} map={similarity.spectrumMapAt(index)} />
</div>
