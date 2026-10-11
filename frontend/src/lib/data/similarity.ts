import type { GetSaliencyData, GetSearchData } from '$lib/api'

export type SimilarityQuery = GetSearchData['query']

export type SaliencyQuery = GetSaliencyData['query']

export type Extent = readonly [number, number]

export interface SimilarityResult {
  request: SimilarityQuery
  galaxies: Int32Array
  similarities: Float32Array
  imageMaps: Float32Array
  spectrumMaps: Float32Array
  imagePositions: (number[] | null)[]
  spectrumPositions: (number[] | null)[]
  tableValueMap: Float32Array
  predicted: string[][]
}

export interface SaliencyResult {
  image: Float32Array
  spectrum: Float32Array
  tableValues: Float32Array
}
