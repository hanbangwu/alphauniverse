<script lang="ts">
  import EnumSelect from '$lib/components/common/enum-select.svelte'
  import FieldSlider from '$lib/components/common/field-slider.svelte'
  import { Switch } from '$lib/components/ui/switch'
  import { getMask, getSearch, getView } from '$lib/state/app.svelte'

  const search = getSearch()
  const view = getView()
  const display = getMask()

  const idle = $derived(view.patches.value.length === 0)
</script>

<EnumSelect field={search.method} label="Method" width="w-44" />

{#if search.blends}
  <label class="flex items-center gap-2 text-sm font-medium">
    <span class="text-secondary-foreground">
      {Math.round((1 - search.weight.value) * 100)}% encoded
    </span>
    <FieldSlider field={search.weight} class="w-40" />
    <span class="text-secondary-foreground">
      {Math.round(search.weight.value * 100)}% codebook
    </span>
  </label>
{/if}

<EnumSelect
  field={search.combine}
  label="Combine"
  width="w-32"
  disabled={view.patches.value.length < 2}
/>

<label class="flex items-center gap-2 text-sm font-medium">
  <Switch
    checked={display.on.value}
    disabled={idle}
    onCheckedChange={(checked) => (display.on.value = checked)}
  />
  Mask
</label>
