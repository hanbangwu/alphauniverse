<script lang="ts">
  import * as NativeSelect from '$lib/components/ui/native-select'
  import { getView } from '$lib/state/app.svelte'

  const view = getView()
</script>

<label class="flex items-center gap-2 text-sm font-medium">
  Data source
  <NativeSelect.Root size="sm" bind:value={view.dataSource.value}>
    <NativeSelect.Option value="dataset">Dataset</NativeSelect.Option>
    <NativeSelect.Option value="prediction">Prediction</NativeSelect.Option>
  </NativeSelect.Root>
</label>
