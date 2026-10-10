<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import { Slider } from '$lib/components/ui/slider'
  import type { Extent } from '$lib/data/similarity'
  import { DECIMALS, type Threshold } from '$lib/state/mask.svelte'

  interface Props {
    domain: Extent
    control: Threshold
  }

  let { domain, control }: Props = $props()

  const threshold = $derived(control.at(domain))
</script>

<div class="flex items-center gap-3">
  <Slider
    type="single"
    value={threshold}
    onValueChange={(value) => (control.override.value = value)}
    min={domain[0]}
    max={domain[1]}
    step={(domain[1] - domain[0]) / 200 || 0.001}
    class="flex-1"
  />
  <span class="w-14 text-right font-mono text-xs tabular-nums">
    {threshold.toFixed(DECIMALS)}
  </span>
  <span class={control.pinned ? undefined : 'cursor-not-allowed'}>
    <Button
      variant="ghost"
      size="sm"
      class="text-xs"
      disabled={!control.pinned}
      onclick={() => control.override.reset()}
    >
      Reset
    </Button>
  </span>
</div>
