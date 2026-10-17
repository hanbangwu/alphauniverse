<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import { errorMessage } from '$lib/errors'
  import { getApp } from '$lib/state/app.svelte'

  interface Props {
    galaxies: number[]
    onclear: () => void
  }

  let { galaxies, onclear }: Props = $props()

  const { view, mosaic } = getApp()

  const SHOWN = 60

  let error = $state<string | null>(null)

  async function download(): Promise<void> {
    try {
      const url = URL.createObjectURL(new Blob([await mosaic.csv(galaxies)], { type: 'text/csv' }))
      const link = document.createElement('a')
      link.href = url
      link.download = 'alphauniverse-selection.csv'
      link.click()
      URL.revokeObjectURL(url)
      error = null
    } catch (caught) {
      error = errorMessage(caught)
    }
  }

  function searchFrom(galaxy: number): void {
    view.select(galaxy)
    view.explorer.value = true
  }
</script>

<section
  class="absolute top-3 right-3 flex w-64 flex-col gap-2 rounded-md border bg-popover/90 p-2 text-xs"
  aria-label="Selected galaxies"
>
  <div class="flex items-center justify-between gap-2">
    <span class="font-medium">{galaxies.length} galaxies selected</span>
    <Button variant="ghost" size="xs" onclick={onclear}>Clear</Button>
  </div>
  {#if galaxies.length}
    <span class="text-muted-foreground">Search from one:</span>
    <div class="flex max-h-40 flex-wrap gap-1 overflow-y-auto">
      {#each galaxies.slice(0, SHOWN) as galaxy (galaxy)}
        <Button
          variant="outline"
          size="xs"
          class="font-mono"
          aria-label="Search from galaxy {galaxy}"
          onclick={() => searchFrom(galaxy)}>#{galaxy}</Button
        >
      {/each}
    </div>
    {#if galaxies.length > SHOWN}
      <span class="text-muted-foreground">The first {SHOWN} are listed.</span>
    {/if}
    <Button variant="secondary" size="xs" onclick={download}>Export properties as CSV</Button>
  {/if}
  {#if error}
    <p role="alert" class="text-destructive">{error}</p>
  {/if}
</section>
