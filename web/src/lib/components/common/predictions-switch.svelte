<script lang="ts">
  import { Switch } from '$lib/components/ui/switch'
  import { getView } from '$lib/state/app.svelte'

  const view = getView()
</script>

<label class="flex items-center gap-2 text-sm font-medium">
  <Switch bind:checked={view.predictions.value} />
  AION predictions
</label>
