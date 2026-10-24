<script lang="ts">
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import { Button } from '$lib/components/ui/button'
  import * as Table from '$lib/components/ui/table'
  import { errorMessage } from '$lib/errors'
  import { getApp } from '$lib/state/app.svelte'

  interface Props {
    galaxies: number[]
    loading: boolean
  }

  let { galaxies, loading }: Props = $props()

  const { view, mosaic } = getApp()

  let error = $state<string | null>(null)

  async function download(): Promise<void> {
    try {
      const url = URL.createObjectURL(new Blob([await mosaic.csv(galaxies)], { type: 'text/csv' }))
      const link = document.createElement('a')
      link.href = url
      link.download = 'alphauniverse-selection.csv'
      link.click()
      setTimeout(() => URL.revokeObjectURL(url))
      error = null
    } catch (caught) {
      error = errorMessage(caught)
    }
  }
</script>

<section
  class="absolute top-3 right-3 flex w-64 flex-col gap-2 rounded-md border bg-popover/90 p-2 text-xs"
  aria-label="Selected galaxies"
>
  <span class="font-medium">{galaxies.length} galaxies selected</span>
  {#if galaxies.length}
    <div class="max-h-40 overflow-y-auto">
      <Table.Root class="text-xs">
        <Table.Header>
          <Table.Row class="hover:bg-transparent">
            <Table.Head class="h-7 px-0 font-normal">Object ID</Table.Head>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {#each galaxies as galaxy (galaxy)}
            <Table.Row class="last:border-0">
              <Table.Cell class="p-0">
                <button
                  type="button"
                  class="w-full cursor-pointer py-1.5 text-left font-mono outline-none aria-pressed:font-semibold focus-visible:ring-2 focus-visible:ring-ring/50"
                  aria-pressed={view.galaxy.value === galaxy}
                  onclick={() => view.select(galaxy)}><GalaxyName {galaxy} /></button
                >
              </Table.Cell>
            </Table.Row>
          {/each}
        </Table.Body>
      </Table.Root>
    </div>
    <Button variant="secondary" size="xs" disabled={loading} onclick={download}
      >Export properties as CSV</Button
    >
  {/if}
  {#if error}
    <p role="alert" class="text-destructive">{error}</p>
  {/if}
</section>
