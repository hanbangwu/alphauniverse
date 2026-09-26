<script lang="ts">
  import { Switch } from '$lib/components/ui/switch'
  import { getMask } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
  const display = getMask()

  const idle = $derived(!similarity.searched)
</script>

<label class="flex items-center gap-2 text-sm font-medium">
  <Switch
    checked={display.on.value}
    disabled={idle}
    onCheckedChange={(checked) => (display.on.value = checked)}
  />
  Mask
</label>
