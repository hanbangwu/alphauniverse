<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import { Slider } from '$lib/components/ui/slider'
  import { Switch } from '$lib/components/ui/switch'
  import type { Extent } from '$lib/data/similarity'
  import { getMask } from '$lib/state/app.svelte'
  import { DECIMALS } from '$lib/state/mask.svelte'

  interface Props {
    domain: Extent
  }

  let { domain }: Props = $props()

  const display = getMask()

  const threshold = $derived(display.at(domain))
</script>

<label class="flex items-center gap-2 text-sm font-medium">
  <Switch
    checked={display.invert.value}
    onCheckedChange={(checked) => (display.invert.value = checked)}
  />
  Invert
</label>

<div class="flex flex-1 items-center gap-3">
  <Slider
    type="single"
    value={threshold}
    onValueChange={(value) => (display.override.value = value)}
    min={domain[0]}
    max={domain[1]}
    step={(domain[1] - domain[0]) / 200 || 0.001}
    class="flex-1"
  />
  <span class="w-14 text-right font-mono text-xs tabular-nums">
    {threshold.toFixed(DECIMALS)}
  </span>
  <Button
    variant="ghost"
    size="sm"
    class="text-xs"
    disabled={!display.pinned}
    onclick={() => display.override.reset()}
  >
    Reset
  </Button>
</div>
