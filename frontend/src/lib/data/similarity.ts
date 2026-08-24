import type { GetSimilarityData } from '$lib/api'

export type SimilarityQuery = GetSimilarityData['query']

export type Extent = readonly [number, number]

export interface SimilarityResult {
  galaxies: Int32Array
  scores: Float32Array
  encoded: Float32Array
  codebook: Float32Array
}
