import {
  getCosines,
  getGalaxy,
  getImageTokens,
  getMeta,
  getSaliency,
  getSearch,
  getSpectrum,
  getSpectrumTemplate,
  getSpectrumTokens,
  getTable,
  getTextSearch
} from '$lib/api'
import type { CosineMode, ImageSurvey, Projection, SpectrumSurvey } from '$lib/client'
import type { Property } from '$lib/labels'
import type { MosaicState } from '$lib/state/mosaic.svelte'
import { byMode } from './modes'
import { matchMean } from './scores'
import type { SaliencyQuery, SaliencyResult, SimilarityQuery, SimilarityResult } from './similarity'
import type { Spectrum } from './spectra'
import { keepPreviousData, queryOptions, skipToken } from '@tanstack/svelte-query'
import { Query, column, eq, isIn, literal } from '@uwdata/mosaic-sql'
import {
  type Float32,
  type Int32,
  type Table,
  type Utf8,
  type Vector,
  tableFromIPC
} from 'apache-arrow'

const FOREVER = { staleTime: Infinity, gcTime: Infinity } as const

export const metaQuery = queryOptions({
  queryKey: ['meta'] as const,
  queryFn: async () => {
    const { data } = await getMeta({ throwOnError: true })
    return data
  },
  ...FOREVER
})

export function tokensQuery(galaxy: number | null, survey: ImageSurvey) {
  return queryOptions({
    queryKey: ['tokens', galaxy, survey] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getImageTokens({ path: { galaxy, survey }, throwOnError: true })
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

async function spectrumOf(data: Blob): Promise<Spectrum> {
  const table = tableFromIPC<{ wavelength: Float32; flux: Float32 }>(
    new Uint8Array(await data.arrayBuffer())
  )
  return {
    wavelength: table.getChild('wavelength')!.toArray(),
    flux: table.getChild('flux')!.toArray()
  }
}

export function spectrumQuery(galaxy: number | null, survey: SpectrumSurvey) {
  return queryOptions({
    queryKey: ['spectrum', galaxy, survey] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getSpectrum({ path: { galaxy, survey }, throwOnError: true })
            return spectrumOf(data)
          },
    ...FOREVER
  })
}

export function spectrumTemplateQuery(galaxy: number | null, survey: SpectrumSurvey) {
  return queryOptions({
    queryKey: ['spectrum', galaxy, survey, 'template'] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getSpectrumTemplate({
              path: { galaxy, survey },
              throwOnError: true
            })
            return spectrumOf(data)
          },
    ...FOREVER
  })
}

export function spectrumTokensQuery(galaxy: number | null, survey: SpectrumSurvey) {
  return queryOptions({
    queryKey: ['spectrum', galaxy, survey, 'tokens'] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getSpectrumTokens({
              path: { galaxy, survey },
              throwOnError: true
            })
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
        : async ({ signal }) => {
            const { data } = await getCosines({
              path: { galaxy, mode },
              signal,
              throwOnError: true
            })
            return new Float32Array(await data.arrayBuffer())
          },
    staleTime: Infinity
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

export function colouringQuery(
  mosaic: MosaicState,
  points: string | null,
  property: Property | null,
  everywhere: boolean
) {
  return queryOptions({
    queryKey: ['colouring', points, property, everywhere] as const,
    queryFn:
      points === null || property === null
        ? skipToken
        : () => mosaic.colour(points, property, everywhere),
    ...FOREVER
  })
}

export function withinQuery(
  mosaic: MosaicState,
  name: string,
  range: readonly [number, number] | null
) {
  return queryOptions({
    queryKey: ['within', name, range] as const,
    queryFn: range === null ? skipToken : () => mosaic.within(name, range),
    staleTime: Infinity
  })
}

export function valuesQuery(mosaic: MosaicState, name: string | null, galaxies: number[] | null) {
  return queryOptions({
    queryKey: ['values', name, galaxies] as const,
    queryFn:
      name === null || galaxies === null
        ? skipToken
        : async () => {
            const rows = await mosaic.coordinator.query(
              Query.from(name)
                .select('galaxy', 'value')
                .where(isIn(column('galaxy'), galaxies))
            )
            return new Map(
              Array.from(rows, ({ galaxy, value }): [number, number | null] => [galaxy, value])
            )
          },
    staleTime: Infinity
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

export function morphologiesQuery(
  mosaic: MosaicState,
  table: string | null,
  galaxies: number[] | null
) {
  return queryOptions({
    queryKey: ['morphologies', table, galaxies] as const,
    queryFn:
      table === null || galaxies === null
        ? skipToken
        : async () => {
            const rows = await mosaic.coordinator.query(
              Query.from(table)
                .select('galaxy', 'category')
                .where(isIn(column('galaxy'), galaxies))
            )
            return new Map(
              Array.from(rows, ({ galaxy, category }): [number, number] => [galaxy, category])
            )
          },
    staleTime: Infinity
  })
}

function positions(table: Table, mode: string): (number[] | null)[] {
  return Array.from(table.getChild(`${mode}_positions`)!, (row: Vector<Int32> | null) =>
    row ? Array.from(row.toArray()) : null
  )
}

function maps(table: Table, mode: string): Float32Array {
  return table.getChild(mode)!.getChildAt<Float32>(0)!.toArray()
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
            return {
              request,
              galaxies: table.getChild('galaxy')!.toArray(),
              similarities: table.getChild('similarity')!.toArray(),
              maps: byMode((mode) => maps(table, mode)),
              positions: byMode((mode) => positions(table, mode)),
              tableValueMap: matchMean(maps(table, 'table_values'), table.numRows),
              observed: Object.fromEntries(
                table.schema.fields
                  .filter((field) => field.name.startsWith('has_'))
                  .map((field) => [
                    field.name.slice('has_'.length),
                    Array.from(table.getChild(field.name)!, Boolean)
                  ])
              ),
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

export function saliencyQuery(request: SaliencyQuery | null) {
  return queryOptions({
    queryKey: ['saliency', request] as const,
    queryFn:
      request === null
        ? skipToken
        : async ({ signal }): Promise<SaliencyResult> => {
            const { data } = await getSaliency({ query: request, signal, throwOnError: true })
            const table = tableFromIPC(new Uint8Array(await data.arrayBuffer()))
            return {
              maps: byMode((mode) => maps(table, mode)),
              tableValues: maps(table, 'table_values')
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
