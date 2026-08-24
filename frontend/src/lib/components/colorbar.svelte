<script lang="ts">
  import type { Extent } from '$lib/data/similarity'
  import { quantize } from 'd3-interpolate'
  import { scaleLinear } from 'd3-scale'
  import { interpolateViridis } from 'd3-scale-chromatic'

  interface Props {
    domain: Extent
    label?: string
  }

  let { domain, label }: Props = $props()

  const gradient = `linear-gradient(to right, ${quantize(interpolateViridis, 16).join(', ')})`

  const ticks = $derived.by(() => {
    const [low, high] = domain
    const scale = scaleLinear().domain([low, high])
    const format = scale.tickFormat(3)
    const span = high - low || 1
    return scale
      .ticks(3)
      .map((value) => ({ value, label: format(value), offset: ((value - low) / span) * 100 }))
  })
</script>

<div class="flex flex-col gap-1">
  {#if label}
    <span class="text-xs text-muted-foreground">{label}</span>
  {/if}
  <div class="h-2 w-full rounded" style:background={gradient}></div>
  <div class="relative h-4 font-mono text-[10px] text-muted-foreground tabular-nums">
    {#each ticks as { value, label: text, offset } (value)}
      <span class="absolute -translate-x-1/2" style:left={`${offset}%`}>{text}</span>
    {/each}
  </div>
</div>
