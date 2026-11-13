<script lang="ts">
  import GalaxyName from '$lib/components/galaxy-name.svelte'
  import { Button } from '$lib/components/ui/button'
  import * as Table from '$lib/components/ui/table'
  import { coverageQuery } from '$lib/data/queries'
  import { getApp } from '$lib/state/app.svelte'
  import { createQueries } from '@tanstack/svelte-query'
  import { csvFormat } from 'd3-dsv'

  interface Props {
    galaxies: number[]
    loading: boolean
  }

  let { galaxies, loading }: Props = $props()

  const { view } = getApp()

  const coverage = createQueries(() => ({
    queries: galaxies.map((galaxy) => coverageQuery(galaxy))
  }))
  const rows = $derived(coverage.map(({ data }) => data))
  const csv = $derived(
    rows.every((row) => row !== undefined)
      ? `data:text/csv;charset=utf-8,${encodeURIComponent(csvFormat(rows, ['object_id', 'ra', 'dec']))}`
      : undefined
  )
  const error = $derived(coverage.find((query) => query.isError)?.error)
</script>

<section
  class="absolute top-3 right-3 flex w-64 flex-col gap-2 glass rounded-xl p-2 text-xs"
  aria-label="Selected galaxies"
>
  <span class="font-medium">{galaxies.length} galaxies selected</span>
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
  <Button
    variant="secondary"
    size="xs"
    disabled={loading || !csv}
    href={csv}
    download="alphauniverse-selection.csv">Export as CSV</Button
  >
  {#if error}
    <p role="alert" class="text-destructive">{error?.message}</p>
  {/if}
</section>
