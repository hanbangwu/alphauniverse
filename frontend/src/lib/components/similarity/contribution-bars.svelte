<script lang="ts">
  import { MODES } from '$lib/labels'
  import { DECIMALS } from '$lib/state/mask.svelte'
  import { getSimilarity } from './similarity.svelte'

  interface Props {
    index: number
  }

  let { index }: Props = $props()

  const similarity = getSimilarity()

  const half = (value: number) => (50 * Math.abs(value)) / similarity.contributionLimit
</script>

<dl class="grid max-w-md grid-cols-[auto_1fr_auto] items-center gap-x-3 gap-y-1 text-xs">
  {#each similarity.contributionModes as mode (mode)}
    {@const value = similarity.contributionAt(mode, index)}
    <dt class="text-muted-foreground">{MODES[mode].label}</dt>
    <dd class="relative h-2 rounded-sm bg-muted" aria-hidden="true">
      <div class="absolute inset-y-0 left-1/2 w-px bg-foreground/40"></div>
      <div
        class={['absolute inset-y-0', value < 0 ? 'bg-destructive' : 'bg-primary']}
        style:left="{value < 0 ? 50 - half(value) : 50}%"
        style:width="{half(value)}%"
      ></div>
    </dd>
    <dd class="text-right font-mono tabular-nums">{value.toFixed(DECIMALS)}</dd>
  {/each}
</dl>
