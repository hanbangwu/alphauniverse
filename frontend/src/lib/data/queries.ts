import { getCoverage, getSimilarity, getSpectrum, getSpectrumTokens, getTokens } from '$lib/api'
import type { Meta, Survey } from '$lib/api'
import { must } from '$lib/invariant'
import type { MosaicState } from '$lib/state/mosaic.svelte'
import type { SimilarityQuery, SimilarityResult } from './similarity'
import type { Spectrum, SpectrumSurvey } from './spectra'
import type { DataTag, DefaultError, QueryKey } from '@tanstack/query-core'
import { type UndefinedInitialDataOptions, queryOptions } from '@tanstack/svelte-query'
import { Query, column, eq, literal } from '@uwdata/mosaic-sql'
import { type Float32, tableFromIPC } from 'apache-arrow'

export type QueryFor<TData, TKey extends QueryKey> = UndefinedInitialDataOptions<
  TData,
  DefaultError,
  TData,
  TKey
> & { queryKey: DataTag<TKey, TData, DefaultError> }

export type TokensQuery = QueryFor<
  Uint32Array<ArrayBuffer>,
  readonly ['tokens', string, number | null]
>
export type CoverageQuery = QueryFor<Survey[], readonly ['coverage', string, number | null]>
export type SpectrumQuery = QueryFor<
  Spectrum,
  readonly ['spectrum', string, number | null, SpectrumSurvey | null]
>
export type SpectrumTokensQuery = QueryFor<
  Uint32Array<ArrayBuffer>,
  readonly ['spectrum', string, number | null, SpectrumSurvey | null, 'tokens']
>
export type PointsQuery = QueryFor<string, readonly ['points', string, string]>
export type MorphologyQuery = QueryFor<
  number | null,
  readonly ['morphology', string, string | null, number | null]
>
export type SimilarityResultQuery = QueryFor<
  SimilarityResult,
  readonly ['similarity', string, SimilarityQuery | null]
>

const FOREVER = { staleTime: Infinity, gcTime: Infinity } as const

export function tokensQuery(meta: Meta, galaxy: number | null): TokensQuery {
  return queryOptions({
    queryKey: ['tokens', meta.revision, galaxy] as const,
    queryFn: async () => {
      const { data } = await getTokens({
        path: { galaxy: must(galaxy, 'the galaxy to fetch tokens for') },
        throwOnError: true
      })
      return new Uint32Array(await data.arrayBuffer())
    },
    enabled: galaxy !== null,
    ...FOREVER
  })
}

export function coverageQuery(meta: Meta, galaxy: number | null): CoverageQuery {
  return queryOptions({
    queryKey: ['coverage', meta.revision, galaxy] as const,
    queryFn: async () => {
      const { data } = await getCoverage({
        path: { galaxy: must(galaxy, 'the galaxy to fetch coverage for') },
        throwOnError: true
      })
      return data
    },
    enabled: galaxy !== null,
    ...FOREVER
  })
}

export function spectrumQuery(
  meta: Meta,
  galaxy: number | null,
  survey: SpectrumSurvey | null
): SpectrumQuery {
  return queryOptions({
    queryKey: ['spectrum', meta.revision, galaxy, survey] as const,
    queryFn: async () => {
      const { data } = await getSpectrum({
        path: {
          galaxy: must(galaxy, 'the galaxy to fetch a spectrum for'),
          survey: must(survey, 'the survey to fetch a spectrum from')
        },
        throwOnError: true
      })
      const table = tableFromIPC<{ wavelength: Float32; flux: Float32 }>(
        new Uint8Array(await data.arrayBuffer())
      )
      return {
        wavelength: must(table.getChild('wavelength'), 'the wavelength column').toArray(),
        flux: must(table.getChild('flux'), 'the flux column').toArray()
      }
    },
    enabled: galaxy !== null && survey !== null,
    ...FOREVER
  })
}

export function spectrumTokensQuery(
  meta: Meta,
  galaxy: number | null,
  survey: SpectrumSurvey | null
): SpectrumTokensQuery {
  return queryOptions({
    queryKey: ['spectrum', meta.revision, galaxy, survey, 'tokens'] as const,
    queryFn: async () => {
      const { data } = await getSpectrumTokens({
        path: {
          galaxy: must(galaxy, 'the galaxy to fetch spectrum tokens for'),
          survey: must(survey, 'the survey to fetch spectrum tokens from')
        },
        throwOnError: true
      })
      return new Uint32Array(await data.arrayBuffer())
    },
    enabled: galaxy !== null && survey !== null,
    ...FOREVER
  })
}

export function pointsQuery(
  mosaic: MosaicState,
  meta: Meta,
  role: string,
  enabled = true
): PointsQuery {
  return queryOptions({
    queryKey: ['points', meta.revision, role] as const,
    queryFn: () => mosaic.load(role),
    enabled,
    ...FOREVER
  })
}

export function morphologyQuery(
  mosaic: MosaicState,
  meta: Meta,
  table: string | null,
  galaxy: number | null
): MorphologyQuery {
  return queryOptions({
    queryKey: ['morphology', meta.revision, table, galaxy] as const,
    queryFn: async () => {
      const rows = await mosaic.coordinator.query(
        Query.from(must(table, 'the point table to read morphology from'))
          .select('category')
          .where(eq(column('galaxy'), literal(must(galaxy, 'the galaxy to read morphology for'))))
      )
      const category = rows.get(0)?.category
      return category == null ? null : Number(category)
    },
    enabled: table !== null && galaxy !== null,
    ...FOREVER
  })
}

export function similarityQuery(
  meta: Meta,
  request: SimilarityQuery | null
): SimilarityResultQuery {
  return queryOptions({
    queryKey: ['similarity', meta.revision, request] as const,
    queryFn: async ({ signal }) => {
      const { data } = await getSimilarity({
        query: must(request, 'the similarity request'),
        signal,
        throwOnError: true
      })
      const table = tableFromIPC(new Uint8Array(await data.arrayBuffer()))
      if (table.batches.length !== 1) {
        throw new Error(`expected one record batch, got ${table.batches.length}`)
      }
      return {
        galaxies: must(table.getChild('galaxy'), 'the similarity galaxy column').toArray(),
        scores: must(table.getChild('score'), 'the similarity score column').toArray(),
        imageMaps: must(
          table.getChild('map')?.getChildAt<Float32>(0),
          'the similarity image map column'
        ).toArray(),
        spectrumMaps: must(
          table.getChild('spectrum')?.getChildAt<Float32>(0),
          'the similarity spectrum map column'
        ).toArray()
      }
    },
    enabled: request !== null,
    staleTime: 5 * 60 * 1000
  })
}
