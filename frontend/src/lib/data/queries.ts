import { getGalaxy, getImageTokens, getSearch, getSpectrum, getSpectrumTokens } from '$lib/api'
import type { Survey } from '$lib/api'
import type { MosaicState } from '$lib/state/mosaic.svelte'
import type { SimilarityQuery, SimilarityResult } from './similarity'
import type { Spectrum } from './spectra'
import type { DataTag, DefaultError, QueryKey } from '@tanstack/query-core'
import {
  type UndefinedInitialDataOptions,
  keepPreviousData,
  queryOptions,
  skipToken
} from '@tanstack/svelte-query'
import { Query, column, eq, literal } from '@uwdata/mosaic-sql'
import { type Float32, tableFromIPC } from 'apache-arrow'

export type QueryFor<TData, TKey extends QueryKey> = UndefinedInitialDataOptions<
  TData,
  DefaultError,
  TData,
  TKey
> & { queryKey: DataTag<TKey, TData, DefaultError> }

export type TokensQuery = QueryFor<Uint32Array<ArrayBuffer>, readonly ['tokens', number | null]>
export type CoverageQuery = QueryFor<Survey[], readonly ['coverage', number | null]>
export type SpectrumQuery = QueryFor<Spectrum, readonly ['spectrum', number | null]>
export type SpectrumTokensQuery = QueryFor<
  Uint32Array<ArrayBuffer>,
  readonly ['spectrum', number | null, 'tokens']
>
export type PointsQuery = QueryFor<string, readonly ['points', string]>
export type MorphologyQuery = QueryFor<
  number | null,
  readonly ['morphology', string | null, number | null]
>
export type SimilarityResultQuery = QueryFor<
  SimilarityResult,
  readonly ['similarity', SimilarityQuery | null]
>

const FOREVER = { staleTime: Infinity, gcTime: Infinity } as const

export function tokensQuery(galaxy: number | null): TokensQuery {
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

export function coverageQuery(galaxy: number | null): CoverageQuery {
  return queryOptions({
    queryKey: ['coverage', galaxy] as const,
    queryFn:
      galaxy === null
        ? skipToken
        : async () => {
            const { data } = await getGalaxy({ path: { galaxy }, throwOnError: true })
            return data.coverage
          },
    ...FOREVER
  })
}

export function spectrumQuery(galaxy: number | null): SpectrumQuery {
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

export function spectrumTokensQuery(galaxy: number | null): SpectrumTokensQuery {
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

export function pointsQuery(mosaic: MosaicState, role: string, enabled = true): PointsQuery {
  return queryOptions({
    queryKey: ['points', role] as const,
    queryFn: () => mosaic.load(role),
    enabled,
    ...FOREVER
  })
}

export function morphologyQuery(
  mosaic: MosaicState,
  table: string | null,
  galaxy: number | null
): MorphologyQuery {
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

export function similarityQuery(request: SimilarityQuery | null): SimilarityResultQuery {
  return queryOptions({
    queryKey: ['similarity', request] as const,
    queryFn:
      request === null
        ? skipToken
        : async ({ signal }) => {
            const { data } = await getSearch({ query: request, signal, throwOnError: true })
            const table = tableFromIPC(new Uint8Array(await data.arrayBuffer()))
            return {
              galaxies: table.getChild('galaxy')!.toArray(),
              scores: table.getChild('score')!.toArray(),
              imageMaps: table.getChild('map')!.getChildAt<Float32>(0)!.toArray(),
              spectrumMaps: table.getChild('spectrum')!.getChildAt<Float32>(0)!.toArray()
            }
          },
    placeholderData: keepPreviousData,
    staleTime: 5 * 60 * 1000
  })
}
