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

  const groups = $derived([...Map.groupBy(table.data ?? [], (row) => row.section)])
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
    return picked ? `${style}; outline: 1px solid ${SELECTED}; outline-offset: -1px` : style
  }

  function format(value: TableRow['value']): string {
    if (value === null) return '-'
    if (typeof value === 'number' && !Number.isInteger(value)) return value.toPrecision(6)
    return String(value)
  }
</script>

<div class="max-h-96 w-full overflow-y-auto rounded-lg border px-3 md:max-w-2xl">
  {#if table.isPending}
    <Spinner class="my-3 size-4 text-muted-foreground" />
  {:else if table.isError}
    <p role="alert" class="py-3 text-xs leading-relaxed text-destructive">
      {errorMessage(table.error)}
    </p>
  {:else}
    <Table.Root class="text-xs">
      <Table.Header>
        <Table.Row class="hover:bg-transparent">
          <Table.Head class="h-7 w-8 px-0 font-normal"
            ><span class="sr-only">Query</span></Table.Head
          >
          <Table.Head class="h-7 px-0 font-normal">Name</Table.Head>
          <Table.Head class="h-7 px-0 text-right font-normal">Value</Table.Head>
          {#if view.predictions.value}
            <Table.Head class="h-7 pr-0 pl-3 text-right font-normal">AION</Table.Head>
          {/if}
        </Table.Row>
      </Table.Header>
      <Table.Body>
        {#each groups as [section, rows] (section)}
          <Table.Row class="hover:bg-transparent">
            <Table.Cell
              colspan={view.predictions.value ? 4 : 3}
              class="px-0 pt-3 pb-1 text-muted-foreground uppercase"
            >
              {@const mode = TABLE_MODES[section]}
              {#if mode}
                <WholeMode {mode}>{SECTIONS[section].label}</WholeMode>
              {:else}
                {SECTIONS[section].label}
              {/if}
            </Table.Cell>
          </Table.Row>
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
        {/each}
      </Table.Body>
    </Table.Root>
  {/if}
</div>
