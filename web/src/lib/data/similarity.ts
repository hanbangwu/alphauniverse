import type { GetModeSearchData, GetSaliencyData, GetSearchData } from '$lib/api'
import type { GalaxyId, TokenMode } from '$lib/client'

export type SimilarityQuery = GetSearchData['query']

export type ModeQuery = GetModeSearchData['query']

export type ModeName = ModeQuery['modes'][number]

export type SearchQuery = SimilarityQuery | ModeQuery

export type SaliencyQuery = GetSaliencyData['query']

export type Extent = readonly [number, number]

export const isModeQuery = (query: SearchQuery): query is ModeQuery => 'modes' in query

export interface SimilarityResult {
  request: SearchQuery
  galaxies: GalaxyId[]
  scores: Float32Array
  contributions: Partial<Record<ModeName, Float32Array>>
  maps: Record<TokenMode, Float32Array>
  tableValues: Float32Array
  positions: Partial<Record<TokenMode, (number[] | null)[]>>
  tableValueMap: Float32Array
  observed: Record<string, boolean[]>
  predicted: string[][]
}

export interface SaliencyResult {
  maps: Record<TokenMode, Float32Array>
  tableValues: Float32Array
}
