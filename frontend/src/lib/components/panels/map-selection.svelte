<script lang="ts">
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import { Button } from '$lib/components/ui/button'
  import * as Table from '$lib/components/ui/table'
  import { galaxyRows, saveCsv } from '$lib/download'
  import { errorMessage } from '$lib/errors'
  import { getApp } from '$lib/state/app.svelte'
  import { useQueryClient } from '@tanstack/svelte-query'

  interface Props {
    galaxies: number[]
    loading: boolean
  }

  let { galaxies, loading }: Props = $props()

  const { view } = getApp()
  const queryClient = useQueryClient()

  let error = $state<string | null>(null)

  async function download(): Promise<void> {
    try {
      saveCsv('alphauniverse-selection.csv', await galaxyRows(queryClient, galaxies))
      error = null
    } catch (caught) {
      error = errorMessage(caught)
    }
  }
</script>

<section
  class="absolute top-3 right-[calc(var(--right-panel)+0.75rem)] flex w-64 flex-col gap-2 glass rounded-xl p-2 text-xs"
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
      >Export as CSV</Button
    >
  {/if}
  {#if error}
    <p role="alert" class="text-destructive">{error}</p>
  {/if}
</section>
