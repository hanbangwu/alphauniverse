<script lang="ts">
  import type { TableRow } from '$lib/api'
  import { type RGB, SELECTED, tokenAlpha, tokenColors } from '$lib/color'
  import { Spinner } from '$lib/components/ui/spinner'
  import * as Table from '$lib/components/ui/table'
  import { tableQuery } from '$lib/data/queries'
  import { SECTIONS } from '$lib/labels'
  import { getSimilarity } from './similarity.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    selected: number[]
    ontoggle: (index: number) => void
  }

  let { galaxy, selected, ontoggle }: Props = $props()

  const similarity = getSimilarity()
  const table = createQuery(() => tableQuery(galaxy))

  const groups = $derived([...Map.groupBy(table.data ?? [], (row) => row.section)])
  const palette = $derived(
    tokenColors((table.data ?? []).flatMap((row) => (row.token === null ? [] : [row.token])))
  )

  function paint(color: RGB, alpha: number): string {
    return `background-color: rgb(${color.join(' ')} / ${alpha})`
  }

  function fill(row: TableRow): string | undefined {
    if (row.scalar === null || row.token === null) return undefined
    const picked = selected.includes(row.scalar)
    const score = similarity.tableValueMap?.[row.scalar] ?? NaN
    const heat = similarity.tableValueHeat
    const style =
      heat && !Number.isNaN(score)
        ? paint(heat(score), 1)
        : paint(palette(row.token), tokenAlpha(picked))
    return picked ? `${style}; outline: 1px solid ${SELECTED}; outline-offset: -1px` : style
  }

  function format(value: TableRow['value']): string {
    if (value === null) return '-'
    if (typeof value === 'number' && !Number.isInteger(value)) return value.toPrecision(6)
    return String(value)
  }
</script>

<div class="flex flex-1 flex-col gap-2">
  <div class="flex items-center justify-between gap-2">
    <span class="text-sm font-medium">Tabular Data</span>
    {#if table.isPending}
      <Spinner class="size-3 text-muted-foreground" />
    {/if}
  </div>

  <div class="aspect-square w-full overflow-y-auto rounded-lg border px-3">
    {#if table.data}
      <Table.Root class="text-xs">
        <Table.Header>
          <Table.Row class="hover:bg-transparent">
            <Table.Head class="h-7 w-8 px-0 font-normal"
              ><span class="sr-only">Query</span></Table.Head
            >
            <Table.Head class="h-7 px-0 font-normal">Name</Table.Head>
            <Table.Head class="h-7 px-0 text-right font-normal">Value</Table.Head>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          {#each groups as [section, rows] (section)}
            <Table.Row class="hover:bg-transparent">
              <Table.Cell colspan={3} class="px-0 pt-3 pb-1 text-muted-foreground uppercase">
                {SECTIONS[section].label}
              </Table.Cell>
            </Table.Row>
            {#each rows as row (row.column)}
              <Table.Row
                class={[
                  'transition-none last:border-0',
                  ((row.scalar !== null && row.token === null) || row.excluded !== null) &&
                    'cursor-not-allowed',
                  row.excluded !== null && 'text-muted-foreground italic'
                ]}
                style={fill(row)}
              >
                <Table.Cell class="px-0 py-1.5">
                  {#if row.scalar !== null}
                    {@const tableValue = row.scalar}
                    <input
                      type="checkbox"
                      class="accent-primary disabled:cursor-not-allowed"
                      checked={selected.includes(tableValue)}
                      disabled={row.token === null}
                      onchange={() => ontoggle(tableValue)}
                      aria-label={`Query ${row.column}`}
                    />
                  {:else if row.excluded !== null}
                    <input
                      type="checkbox"
                      class="disabled:cursor-not-allowed"
                      disabled
                      aria-label={`Query ${row.column}: ${row.excluded}`}
                    />
                  {/if}
                </Table.Cell>
                <Table.Cell class="px-0 py-1.5 font-mono">
                  {row.column}
                  {#if row.excluded !== null}
                    <span class="block font-sans text-[10px]">{row.excluded}</span>
                  {/if}
                </Table.Cell>
                <Table.Cell class="px-0 py-1.5 text-right font-mono tabular-nums">
                  {format(row.value)}
                </Table.Cell>
              </Table.Row>
            {/each}
          {/each}
        </Table.Body>
      </Table.Root>
    {/if}
  </div>
</div>
