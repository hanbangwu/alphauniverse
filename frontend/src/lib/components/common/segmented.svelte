<script lang="ts" generics="T extends string">
  import { Button } from '$lib/components/ui/button'
  import type { EnumField } from '$lib/state/fields.svelte'

  interface Props {
    field: EnumField<T>
  }

  let { field }: Props = $props()
</script>

<div
  class="grid gap-1"
  style:grid-template-columns={`repeat(${field.choices.length}, minmax(0, 1fr))`}
>
  {#each field.choices as { value, label } (value)}
    <Button
      variant={field.value === value ? 'secondary' : 'ghost'}
      size="sm"
      class="px-1.5 text-xs"
      onclick={() => (field.value = value)}
    >
      {label}
    </Button>
  {/each}
</div>
