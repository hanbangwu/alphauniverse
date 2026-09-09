<script lang="ts">
  import { Switch } from '$lib/components/ui/switch'
  import { getMask, getView } from '$lib/state/app.svelte'

  const view = getView()
  const display = getMask()

  const idle = $derived(view.patches.value.length === 0)
</script>

<label class="flex items-center gap-2 text-sm font-medium">
  <Switch
    checked={display.on.value}
    disabled={idle}
    onCheckedChange={(checked) => (display.on.value = checked)}
  />
  Mask
</label>
