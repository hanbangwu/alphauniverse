<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import { cn } from '$lib/utils'
  import type { Snippet } from 'svelte'

  interface Props {
    children: Snippet
    busy?: boolean
    class?: string
  }

  let { children, busy = false, class: className }: Props = $props()
</script>

<div
  class={cn('relative aspect-square w-full overflow-hidden rounded-lg border bg-black', className)}
>
  {@render children()}
  {#if busy}
    <div class="absolute inset-0 grid place-content-center bg-black/50">
      <Spinner class="size-6" />
    </div>
  {/if}
</div>
