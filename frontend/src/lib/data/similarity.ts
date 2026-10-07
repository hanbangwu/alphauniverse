import type { GetSearchData } from '$lib/api'

export type SimilarityQuery = GetSearchData['query']

export type Extent = readonly [number, number]

export interface SimilarityResult {
  galaxies: Int32Array
  scores: Float32Array
  imageMaps: Float32Array
  spectrumMaps: Float32Array
  scalarMap: Float32Array
}
