import {
  getCosines,
  getGalaxy,
  getImageTokens,
  getMeta,
  getSearch,
  getSpectrum,
  getSpectrumTokens,
  getTable,
  getTextSearch
} from '$lib/api'
import type { CosineMode, Projection } from '$lib/client'
import type { MosaicState } from '$lib/state/mosaic.svelte'
import { matchMean } from './scores'
import type { SimilarityQuery, SimilarityResult } from './similarity'
import { keepPreviousData, queryOptions, skipToken } from '@tanstack/svelte-query'
import { Query, column, eq, literal } from '@uwdata/mosaic-sql'
import { type Float32, type Table, type Utf8, type Vector, tableFromIPC } from 'apache-arrow'

const FOREVER = { staleTime: Infinity, gcTime: Infinity } as const

export const metaQuery = queryOptions({
  queryKey: ['meta'] as const,
  queryFn: async () => {
    const { data } = await getMeta({ throwOnError: true })
    return data
  },
  ...FOREVER
})

export function tokensQuery(galaxy: number | null) {
  return queryOptions({
    queryKey: ['tokens', galaxy] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getImageTokens({ path: { galaxy }, throwOnError: true })
            return new Uint32Array(await data.arrayBuffer())
          },
    ...FOREVER
  })
}

export function coverageQuery(galaxy: number | null) {
  return queryOptions({
    queryKey: ['coverage', galaxy] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getGalaxy({ path: { galaxy }, throwOnError: true })
            return data
          },
    ...FOREVER
  })
}

export function tableQuery(galaxy: number | null) {
  return queryOptions({
    queryKey: ['table', galaxy] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getTable({ path: { galaxy }, throwOnError: true })
            return data
          },
    ...FOREVER
  })
}

export function spectrumQuery(galaxy: number | null) {
  return queryOptions({
    queryKey: ['spectrum', galaxy] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getSpectrum({ path: { galaxy }, throwOnError: true })
            const table = tableFromIPC<{ wavelength: Float32; flux: Float32 }>(
              new Uint8Array(await data.arrayBuffer())
            )
            return {
              wavelength: table.getChild('wavelength')!.toArray(),
              flux: table.getChild('flux')!.toArray()
            }
          },
    ...FOREVER
  })
}

export function spectrumTokensQuery(galaxy: number | null) {
  return queryOptions({
    queryKey: ['spectrum', galaxy, 'tokens'] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getSpectrumTokens({ path: { galaxy }, throwOnError: true })
            return new Uint32Array(await data.arrayBuffer())
          },
    ...FOREVER
  })
}

export function cosinesQuery(galaxy: number | null, mode: CosineMode) {
  return queryOptions({
    queryKey: ['cosines', galaxy, mode] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getCosines({ path: { galaxy, mode }, throwOnError: true })
            return new Float32Array(await data.arrayBuffer())
          },
    ...FOREVER
  })
}

export function pointsQuery(mosaic: MosaicState, projection: Projection, enabled = true) {
  return queryOptions({
    queryKey: ['points', projection] as const,
    queryFn: () => mosaic.load(projection),
    enabled,
    ...FOREVER
  })
}

export function morphologyQuery(mosaic: MosaicState, table: string | null, galaxy: number | null) {
  return queryOptions({
    queryKey: ['morphology', table, galaxy] as const,
    queryFn:
      table === null || galaxy === null
        ? skipToken
        : async () => {
            const rows = await mosaic.coordinator.query(
              Query.from(table)
                .select('category')
                .where(eq(column('galaxy'), literal(galaxy)))
            )
            const category = rows.get(0)?.category
            return category == null ? null : Number(category)
          },
    ...FOREVER
  })
}

function shownMaps(table: Table, mode: string): Float32Array {
  const selection = table.getChild(`${mode}_selection`)!
  const shown = selection.nullCount === 0 ? selection : table.getChild(mode)!
  return shown.getChildAt<Float32>(0)!.toArray()
}

export function similarityQuery(request: SimilarityQuery | null) {
  return queryOptions({
    queryKey: ['similarity', request] as const,
    queryFn:
      request === null
        ? skipToken
        : async ({ signal }): Promise<SimilarityResult> => {
            const { data } = await getSearch({ query: request, signal, throwOnError: true })
            const table = tableFromIPC(new Uint8Array(await data.arrayBuffer()))
            const tableValueMaps = table.getChild('table_values')!.getChildAt<Float32>(0)!.toArray()
            return {
              galaxies: table.getChild('galaxy')!.toArray(),
              similarities: table.getChild('similarity')!.toArray(),
              imageMaps: shownMaps(table, 'ls_image'),
              spectrumMaps: shownMaps(table, 'desi_spectrum'),
              tableValueMap: matchMean(tableValueMaps, table.numRows),
              predicted: Array.from(
                table.getChild('predicted')!,
                (row: Vector<Utf8>) => Array.from(row) as string[]
              )
            }
          },
    placeholderData: keepPreviousData,
    staleTime: Infinity
  })
}

export function textSearchQuery(text: string | null) {
  return queryOptions({
    queryKey: ['text-search', text] as const,
    queryFn:
      text === null
        ? skipToken
        : async ({ signal }) => {
            const { data } = await getTextSearch({ query: { text }, signal, throwOnError: true })
            return data
          },
    placeholderData: keepPreviousData,
    staleTime: Infinity
  })
}
