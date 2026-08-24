<script lang="ts" generics="T extends string">
  import * as Select from '$lib/components/ui/select'
  import type { EnumField } from '$lib/state/fields.svelte'

  interface Props {
    field: EnumField<T>
    label?: string
    disabled?: boolean
    width?: string
  }

  let { field, label = '', disabled = false, width = 'w-40' }: Props = $props()
</script>

<label class="flex items-center gap-2 text-sm font-medium">
  {label}
  <Select.Root
    type="single"
    value={field.value}
    {disabled}
    onValueChange={(value) => (field.value = value as T)}
  >
    <Select.Trigger class={width}>{field.label}</Select.Trigger>
    <Select.Content class="z-200">
      {#each field.choices as { value, label: option } (value)}
        <Select.Item {value} label={option}>{option}</Select.Item>
      {/each}
    </Select.Content>
  </Select.Root>
</label>
