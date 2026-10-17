<script lang="ts">
  import { SELECTED } from '$lib/color'
  import type { ModeName } from '$lib/data/similarity'
  import { MODES } from '$lib/labels'
  import { getView } from '$lib/state/app.svelte'
  import type { Snippet } from 'svelte'

  interface Props {
    mode: ModeName
    children: Snippet
  }

  let { mode, children }: Props = $props()

  const view = getView()
</script>

<div class="relative flex flex-1 flex-col">
  {@render children()}
  {#if view.wholeModes.value}
    {@const selected = view.modes.has(mode)}
    <button
      type="button"
      aria-pressed={selected}
      aria-label="Search on the whole {MODES[mode].label}"
      onclick={() => view.modes.toggle(mode)}
      class="absolute -inset-1 cursor-pointer rounded-lg outline-offset-0 hover:bg-foreground/5 focus-visible:ring-3 focus-visible:ring-ring/50"
      style:outline={selected ? `2px solid ${SELECTED}` : undefined}
    ></button>
  {/if}
</div>
