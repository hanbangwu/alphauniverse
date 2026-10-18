<script lang="ts">
  import type { TableRow } from '$lib/api'
  import { type RGB, SELECTED, tokenAlpha, tokenColors } from '$lib/color'
  import { Spinner } from '$lib/components/ui/spinner'
  import * as Table from '$lib/components/ui/table'
  import { TABLE_MODES } from '$lib/data/modes'
  import { tableQuery } from '$lib/data/queries'
  import { errorMessage } from '$lib/errors'
  import { SECTIONS } from '$lib/labels'
  import { getMask, getView } from '$lib/state/app.svelte'
  import { getSimilarity } from './similarity.svelte'
  import WholeMode from './whole-mode.svelte'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    selected: number[]
    ontoggle: (index: number) => void
  }

  let { galaxy, selected, ontoggle }: Props = $props()

  const similarity = getSimilarity()
  const display = getMask()
  const view = getView()
  const table = createQuery(() => tableQuery(galaxy))

  const order = Object.keys(SECTIONS)
  const groups = $derived(
    [...Map.groupBy(table.data ?? [], (row) => row.section)].sort(
      ([first], [second]) => order.indexOf(first) - order.indexOf(second)
    )
  )

  const mask = $derived(
    similarity.tableValueMap &&
      similarity.maskOf(similarity.tableValueMap, similarity.tableValueDomain, display.table)
  )
  const palette = $derived(
    tokenColors((table.data ?? []).flatMap((row) => (row.token === null ? [] : [row.token])))
  )

  function paint(color: RGB, alpha: number): string {
    return `background-color: rgb(${color.join(' ')} / ${alpha})`
  }

  function chosen(row: TableRow): boolean {
    if (!view.wholeModes.value)
      return row.table_value !== null && selected.includes(row.table_value)
    const mode = TABLE_MODES[row.section]
    return mode !== undefined && view.modes.has(mode)
  }

  function fill(row: TableRow): string | undefined {
    if (row.table_value === null || row.token === null) return undefined
    const picked = chosen(row)
    const score = similarity.tableValueMap?.[row.table_value] ?? NaN
    const heat = similarity.tableValueHeat
    const style =
      heat && !Number.isNaN(score)
        ? paint(heat(score), mask?.[row.table_value] === 0 ? 0 : 1)
        : paint(palette(row.token), tokenAlpha(picked))
    return picked && !view.wholeModes.value
      ? `${style}; outline: 1px solid ${SELECTED}; outline-offset: -1px`
      : style
  }

  function format(value: TableRow['value']): string {
    if (value === null) return '-'
    if (typeof value === 'number' && !Number.isInteger(value)) return value.toPrecision(6)
    return String(value)
  }
</script>

{#snippet sectionTable(section: TableRow['section'], rows: TableRow[])}
  <div class="max-h-96 overflow-y-auto rounded-lg border px-3">
    <Table.Root class="text-xs" aria-label={SECTIONS[section].label}>
      <Table.Header>
        <Table.Row class="hover:bg-transparent">
          <Table.Head class="h-7 w-8 px-0 font-normal"
            ><span class="sr-only">Query</span></Table.Head
          >
          <Table.Head class="h-7 px-0 font-normal text-muted-foreground uppercase">
            <span aria-hidden="true">{SECTIONS[section].label}</span>
            <span class="sr-only">Name</span>
          </Table.Head>
          <Table.Head class="h-7 px-0 text-right font-normal">Value</Table.Head>
          {#if view.predictions.value}
            <Table.Head class="h-7 pr-0 pl-3 text-right font-normal">AION</Table.Head>
          {/if}
        </Table.Row>
      </Table.Header>
      <Table.Body>
        {#each rows as row (row.column)}
          <Table.Row
            class={[
              'transition-none last:border-0',
              ((row.table_value !== null && row.token === null) || row.excluded !== null) &&
                'cursor-not-allowed',
              row.excluded !== null && 'text-muted-foreground italic'
            ]}
            style={fill(row)}
          >
            <Table.Cell class="px-0 py-1.5">
              {#if row.table_value !== null}
                {@const tableValue = row.table_value}
                <input
                  type="checkbox"
                  class="accent-primary disabled:cursor-not-allowed"
                  checked={chosen(row)}
                  disabled={row.token === null || view.wholeModes.value}
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
            {#if view.predictions.value}
              <Table.Cell class="py-1.5 pr-0 pl-3 text-right font-mono tabular-nums">
                {format(row.template)}
              </Table.Cell>
            {/if}
          </Table.Row>
        {/each}
      </Table.Body>
    </Table.Root>
  </div>
{/snippet}

{#if table.isPending}
  <Spinner class="my-3 size-4 text-muted-foreground" />
{:else if table.isError}
  <p role="alert" class="py-3 text-xs leading-relaxed text-destructive">
    {errorMessage(table.error)}
  </p>
{:else}
  <div class="grid w-full items-start gap-5 md:grid-cols-2">
    {#each groups as [section, rows] (section)}
      {@const mode = TABLE_MODES[section]}
      {#if mode}
        <WholeMode {mode}>{@render sectionTable(section, rows)}</WholeMode>
      {:else}
        {@render sectionTable(section, rows)}
      {/if}
    {/each}
  </div>
{/if}
