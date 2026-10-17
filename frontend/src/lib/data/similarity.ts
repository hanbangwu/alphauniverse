import type { GetSaliencyData, GetSearchData } from '$lib/api'
import type { CosineMode } from '$lib/client'

export type SimilarityQuery = GetSearchData['query']

export type SaliencyQuery = GetSaliencyData['query']

export type Extent = readonly [number, number]

export interface SimilarityResult {
  request: SimilarityQuery
  galaxies: Int32Array
  similarities: Float32Array
  maps: Record<CosineMode, Float32Array>
  positions: Record<CosineMode, (number[] | null)[]>
  tableValueMap: Float32Array
  observed: Record<string, boolean[]>
  predicted: string[][]
}

export interface SaliencyResult {
  maps: Record<CosineMode, Float32Array>
  tableValues: Float32Array
}
