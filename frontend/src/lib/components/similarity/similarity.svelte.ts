import { type RGB, continuous } from '$lib/color'
import { cosinesQuery, saliencyQuery, similarityQuery } from '$lib/data/queries'
import { best, extent, mask, row } from '$lib/data/scores'
import type {
  Extent,
  SaliencyResult,
  SimilarityQuery,
  SimilarityResult
} from '$lib/data/similarity'
import { errorMessage } from '$lib/errors'
import { must } from '$lib/invariant'
import { SIMILARITY } from '$lib/labels'
import { type AppState, getApp } from '$lib/state/app.svelte'
import { DECIMALS, type Threshold } from '$lib/state/mask.svelte'
import { type CreateQueryResult, createQuery } from '@tanstack/svelte-query'
import { Context } from 'runed'

type Caption = (value: number, index: number) => string

export type Layer = 'galaxy' | 'population'

export class Similarity {
  readonly galaxy: number
  readonly grid: number

  readonly #app: AppState = getApp()
  readonly #imageTokenCount: number = this.#app.meta.grid ** 2
  readonly #result: CreateQueryResult<SimilarityResult>
  readonly #imageCosines: CreateQueryResult<Float32Array>
  readonly #spectrumCosines: CreateQueryResult<Float32Array>
  readonly #saliency: CreateQueryResult<SaliencyResult>
  #submitted: SimilarityQuery | null = $state(null)

  layer: Layer = $state('galaxy')

  readonly draft: SimilarityQuery | null

  readonly imageMaps: Float32Array | null
  readonly galaxies: Int32Array
  readonly imageDomain: Extent | null
  readonly imageHeat: ((value: number) => RGB) | null
  readonly imageLayer: Float32Array | null
  readonly imageLayerDomain: Extent | null
  readonly imageLayerHeat: ((value: number) => RGB) | null
  readonly spectrumMaps: Float32Array | null
  readonly spectrumHeat: ((value: number) => RGB) | null
  readonly spectrumLayer: Float32Array | null
  readonly spectrumLayerHeat: ((value: number) => RGB) | null
  readonly tableValueMap: Float32Array | null
  readonly tableValueHeat: ((value: number) => RGB) | null

  constructor(galaxy: number) {
    const app = this.#app
    this.galaxy = galaxy
    this.grid = app.meta.grid

    this.draft = $derived(
      app.search.request(
        galaxy,
        app.view.imageTokens.value,
        app.view.spectrumTokens.value,
        app.view.tableValues.value,
        app.view.imageAnywhere.value,
        app.view.spectrumAnywhere.value
      )
    )
    this.#result = createQuery(() => similarityQuery(this.#submitted))
    const own = $derived(this.layer === 'galaxy')
    this.#imageCosines = createQuery(() =>
      cosinesQuery(own && app.view.imageTokens.value.length ? galaxy : null, 'ls_image')
    )
    this.#spectrumCosines = createQuery(() =>
      cosinesQuery(own && app.view.spectrumTokens.value.length ? galaxy : null, 'desi_spectrum')
    )
    this.#saliency = createQuery(() =>
      saliencyQuery(
        !own && this.draft
          ? {
              galaxy,
              ls_image: this.draft.ls_image,
              desi_spectrum: this.draft.desi_spectrum,
              table_values: this.draft.table_values,
              anywhere: this.draft.anywhere
            }
          : null
      )
    )
    const saliency = $derived(own || !this.draft ? null : (this.#saliency.data ?? null))

    this.imageMaps = $derived(this.#result.data?.imageMaps ?? null)
    this.galaxies = $derived(this.#result.data?.galaxies ?? new Int32Array())
    this.imageDomain = $derived(
      this.imageMaps ? extent(this.imageMaps.subarray(this.#imageTokenCount)) : null
    )
    this.imageHeat = $derived(this.imageDomain ? continuous(this.imageDomain) : null)
    this.imageLayer = $derived(
      saliency
        ? saliency.image
        : this.#imageCosines.data
          ? best(this.#imageCosines.data, app.view.imageTokens.value)
          : null
    )
    this.imageLayerDomain = $derived(this.imageLayer ? extent(this.imageLayer) : null)
    this.imageLayerHeat = $derived(this.imageLayerDomain ? continuous(this.imageLayerDomain) : null)
    this.spectrumMaps = $derived(this.#result.data?.spectrumMaps ?? null)
    this.spectrumHeat = $derived.by(() => {
      if (!this.spectrumMaps) return null
      const [low, high] = extent(this.spectrumMaps.subarray(this.spectrumMapAt(0).length))
      return low <= high ? continuous([low, high]) : null
    })
    this.spectrumLayer = $derived(
      saliency
        ? saliency.spectrum
        : this.#spectrumCosines.data
          ? best(this.#spectrumCosines.data, app.view.spectrumTokens.value)
          : null
    )
    this.spectrumLayerHeat = $derived(
      this.spectrumLayer ? continuous(extent(this.spectrumLayer)) : null
    )
    this.tableValueMap = $derived(
      saliency ? saliency.tableValues : (this.#result.data?.tableValueMap ?? null)
    )
    this.tableValueHeat = $derived.by(() => {
      if (!this.tableValueMap) return null
      const [low, high] = extent(this.tableValueMap)
      return low <= high ? continuous([low, high]) : null
    })
  }

  get error(): string | null {
    return this.#result.isError ? errorMessage(this.#result.error) : null
  }

  get imageLayerError(): string | null {
    const failed = this.layer === 'galaxy' ? this.#imageCosines : this.#saliency
    return failed.isError ? errorMessage(failed.error) : null
  }

  get spectrumLayerError(): string | null {
    if (this.layer !== 'galaxy') return null
    return this.#spectrumCosines.isError ? errorMessage(this.#spectrumCosines.error) : null
  }

  anywhere(mode: 'ls_image' | 'desi_spectrum'): boolean {
    return this.#result.data?.request.anywhere?.includes(mode) ?? false
  }

  imageOutlinesAt(index: number): number[] {
    const data = this.#result.data
    return data?.imagePositions[index] ?? data?.request.ls_image ?? []
  }

  spectrumOutlinesAt(index: number): number[] {
    const data = this.#result.data
    return data?.spectrumPositions[index] ?? data?.request.desi_spectrum ?? []
  }

  get stale(): boolean {
    return this.#result.isPlaceholderData
  }

  get searched(): boolean {
    return this.#submitted !== null
  }

  submit(): void {
    this.#submitted = this.draft
  }

  get matches(): { galaxy: number; index: number }[] {
    return Array.from(this.galaxies.subarray(1), (galaxy, at) => ({
      galaxy,
      index: at + 1
    }))
  }

  readonly score = (value: number): string =>
    `${SIMILARITY.short} ${Math.min(1, value).toFixed(DECIMALS)}`

  readonly cosine: Caption = (value) => `cosine ${value.toFixed(DECIMALS)}`

  readonly correlation: Caption = (value) => `partial correlation ${value.toFixed(DECIMALS)}`

  get layerCaption(): Caption {
    return this.layer === 'galaxy' ? this.cosine : this.correlation
  }

  readonly caption: Caption = (value) => `token ${value}`

  readonly maskColor = (value: number): RGB => (value ? [255, 255, 255] : [0, 0, 0])

  readonly maskTitle: Caption = (value) => (value ? 'masked' : 'unmasked')

  predictedAt(index: number): string[] {
    return this.#result.data?.predicted[index] ?? []
  }

  scoreAt(index: number): number {
    return this.#result.data?.similarities[index] ?? 0
  }

  imageMapAt(index: number): Float32Array {
    return row(must(this.imageMaps, 'the image maps'), index, this.#imageTokenCount)
  }

  spectrumMapAt(index: number): Float32Array {
    const maps = must(this.spectrumMaps, 'the spectrum maps')
    return row(maps, index, maps.length / this.galaxies.length)
  }

  maskOf(values: ArrayLike<number>, domain: Extent | null, control: Threshold): Uint8Array | null {
    const display = this.#app.mask
    if (!display.on.value || !domain) return null
    return mask(values, control.at(domain), display.invert.value)
  }
}

const context = new Context<Similarity>('similarity')

export const setSimilarity = (value: Similarity): Similarity => context.set(value)
export const getSimilarity = (): Similarity => context.get()
