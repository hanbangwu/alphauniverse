<script lang="ts">
  import type { CosineMode } from '$lib/client'
  import ImageTokenPanel from './image-token-panel.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    mode: CosineMode
    label: string
    values: Float32Array
    selected: number[]
  }

  let { mode, label, values, selected }: Props = $props()

  const similarity = getSimilarity()
</script>

<ImageTokenPanel
  {label}
  {values}
  grid={similarity.grid}
  color={similarity.heats?.[mode] ?? null}
  title={similarity.score}
  {selected}
/>
