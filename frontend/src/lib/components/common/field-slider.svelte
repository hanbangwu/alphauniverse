<script lang="ts">
  import { Slider } from '$lib/components/ui/slider'
  import type { NumberField } from '$lib/state/fields.svelte'

  interface Props {
    field: NumberField
  }

  let { field }: Props = $props()
</script>

<Slider
  type="single"
  value={field.value}
  onValueChange={(value) => (field.value = value)}
  min={field.minimum}
  max={field.maximum}
  step={field.step}
  class="w-32"
/>
