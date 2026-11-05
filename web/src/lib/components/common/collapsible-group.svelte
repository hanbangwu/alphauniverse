<script lang="ts">
  import ChevronRightIcon from '@lucide/svelte/icons/chevron-right'
  import { Collapsible } from 'bits-ui'
  import type { Snippet } from 'svelte'

  interface Props {
    title: string
    open: boolean
    actions?: Snippet
    children: Snippet
  }

  let { title, open = $bindable(), actions, children }: Props = $props()
</script>

<Collapsible.Root bind:open class="flex flex-col gap-3">
  <div class="flex flex-wrap items-center gap-x-4 gap-y-1">
    <Collapsible.Trigger class="flex items-center gap-1 text-sm font-medium">
      <ChevronRightIcon class="size-4 transition-transform {open ? 'rotate-90' : ''}" />
      {title}
    </Collapsible.Trigger>
    {@render actions?.()}
  </div>
  <Collapsible.Content class="flex flex-col gap-5">
    {#if open}
      {@render children()}
    {/if}
  </Collapsible.Content>
</Collapsible.Root>
