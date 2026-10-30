import { coverageQuery } from '$lib/data/queries'
import type { QueryClient } from '@tanstack/svelte-query'
import { csvFormat } from 'd3-dsv'

export interface EcsvColumn {
  name: string
  datatype: 'string' | 'float32' | 'float64'
  subtype?: string
  unit?: string
}

export interface Ecsv {
  columns: EcsvColumn[]
  rows: Record<string, unknown>[]
  meta: Record<string, unknown>
}

export function galaxyRows(queryClient: QueryClient, galaxies: number[]) {
  return Promise.all(
    galaxies.map(async (galaxy) => {
      const { object_id, ra, dec } = await queryClient.fetchQuery(coverageQuery(galaxy))
      return { object_id, ra, dec }
    })
  )
}

function save(filename: string, text: string, type: string): void {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  setTimeout(() => URL.revokeObjectURL(url))
}

export function saveCsv(filename: string, rows: Record<string, unknown>[]): void {
  save(filename, csvFormat(rows), 'text/csv')
}

export function saveEcsv(filename: string, { columns, rows, meta }: Ecsv): void {
  const header = [
    '%ECSV 1.0',
    '---',
    "delimiter: ','",
    'datatype:',
    ...columns.map((column) => `- ${JSON.stringify(column)}`),
    `meta: ${JSON.stringify(meta)}`
  ]
  const body = csvFormat(
    rows,
    columns.map(({ name }) => name)
  )
  save(filename, [...header.map((line) => `# ${line}`), body].join('\n'), 'text/plain')
}
