import { type RGB, continuous } from '$lib/color'
import { similarityQuery } from '$lib/data/queries'
import { extent, mask, row } from '$lib/data/scores'
import type { Extent, SimilarityQuery, SimilarityResult } from '$lib/data/similarity'
import { errorMessage } from '$lib/errors'
import { must } from '$lib/invariant'
import { SIMILARITY } from '$lib/labels'
import { type AppState, getApp } from '$lib/state/app.svelte'
import { DECIMALS } from '$lib/state/mask.svelte'
import { type CreateQueryResult, createQuery } from '@tanstack/svelte-query'
import { Context } from 'runed'

type Caption = (value: number, index: number) => string

export class Similarity {
  readonly galaxy: number
  readonly grid: number

  readonly #app: AppState = getApp()
  readonly #imageTokenCount: number = this.#app.meta.grid ** 2
  readonly #result: CreateQueryResult<SimilarityResult>
  #submitted: SimilarityQuery | null = $state(null)

  readonly draft: SimilarityQuery | null

  readonly imageMaps: Float32Array | null
  readonly galaxies: Int32Array
  readonly imageDomain: Extent | null
  readonly imageHeat: ((value: number) => RGB) | null
  readonly imageMap: Float32Array | null
  readonly spectrumMaps: Float32Array | null
  readonly spectrumHeat: ((value: number) => RGB) | null
  readonly spectrumMap: Float32Array | null
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
        app.view.tableValues.value
      )
    )
    this.#result = createQuery(() => similarityQuery(this.#submitted))

    this.imageMaps = $derived(this.#result.data?.imageMaps ?? null)
    this.galaxies = $derived(this.#result.data?.galaxies ?? new Int32Array())
    this.imageDomain = $derived(this.imageMaps ? extent(this.imageMaps) : null)
    this.imageHeat = $derived(this.imageDomain ? continuous(this.imageDomain) : null)
    this.imageMap = $derived(this.imageMaps ? row(this.imageMaps, 0, this.#imageTokenCount) : null)
    this.spectrumMaps = $derived(this.#result.data?.spectrumMaps ?? null)
    this.spectrumHeat = $derived.by(() => {
      if (!this.spectrumMaps) return null
      const [low, high] = extent(this.spectrumMaps)
      return low <= high ? continuous([low, high]) : null
    })
    this.spectrumMap = $derived(this.spectrumMaps ? this.spectrumMapAt(0) : null)
    this.tableValueMap = $derived(this.#result.data?.tableValueMap ?? null)
    this.tableValueHeat = $derived.by(() => {
      if (!this.tableValueMap) return null
      const [low, high] = extent(this.tableValueMap)
      return low <= high ? continuous([low, high]) : null
    })
  }

  get error(): string | null {
    return this.#result.isError ? errorMessage(this.#result.error) : null
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

  readonly overlap: Caption = (value) => `overlap ${Math.exp(value).toPrecision(DECIMALS)}`

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

  maskOf(values: ArrayLike<number>): Uint8Array | null {
    const display = this.#app.mask
    if (!display.on.value || !this.imageDomain) return null
    return mask(values, display.at(this.imageDomain), display.invert.value)
  }
}

const context = new Context<Similarity>('similarity')

export const setSimilarity = (value: Similarity): Similarity => context.set(value)
export const getSimilarity = (): Similarity => context.get()
