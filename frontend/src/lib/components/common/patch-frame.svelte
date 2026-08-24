<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import type { Snippet } from 'svelte'

  interface Props {
    children: Snippet
    busy?: boolean
  }

  let { children, busy = false }: Props = $props()
</script>

<div class="relative aspect-square w-full overflow-hidden rounded-lg border bg-black">
  {@render children()}
  {#if busy}
    <div class="absolute inset-0 grid place-content-center bg-black/50">
      <Spinner class="size-6" />
    </div>
  {/if}
</div>
