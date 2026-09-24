import { type RGB, continuous, tokenColors } from '$lib/color'
import { similarityQuery, tokensQuery } from '$lib/data/queries'
import { extent, mask, row } from '$lib/data/scores'
import type { Extent, SimilarityResult } from '$lib/data/similarity'
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
  readonly #patchCount: number = this.#app.meta.grid ** 2
  readonly #tokenMap: CreateQueryResult<Uint32Array<ArrayBuffer>>
  readonly #result: CreateQueryResult<SimilarityResult>

  readonly imageMaps: Float32Array | null
  readonly galaxies: Int32Array
  readonly imageDomain: Extent | null
  readonly imageHeat: ((value: number) => RGB) | null
  readonly imageMap: Float32Array | null
  readonly spectrumMaps: Float32Array | null
  readonly spectrumHeat: ((value: number) => RGB) | null
  readonly spectrumMap: Float32Array | null
  readonly tokens: Uint32Array | null
  readonly palette: ((value: number) => RGB) | null

  constructor(galaxy: number) {
    const app = this.#app
    this.galaxy = galaxy
    this.grid = app.meta.grid

    this.#tokenMap = createQuery(() => tokensQuery(app.meta, galaxy))
    this.#result = createQuery(() =>
      similarityQuery(
        app.meta,
        app.search.request(galaxy, app.view.patches.value, app.view.spans.value)
      )
    )

    this.imageMaps = $derived(this.#result.data?.imageMaps ?? null)
    this.galaxies = $derived(this.#result.data?.galaxies ?? new Int32Array())
    this.imageDomain = $derived(this.imageMaps ? extent(this.imageMaps) : null)
    this.imageHeat = $derived(this.imageDomain ? continuous(this.imageDomain) : null)
    this.imageMap = $derived(this.imageMaps ? row(this.imageMaps, 0, this.#patchCount) : null)
    this.spectrumMaps = $derived(this.#result.data?.spectrumMaps ?? null)
    this.spectrumHeat = $derived.by(() => {
      if (!this.spectrumMaps) return null
      const [low, high] = extent(this.spectrumMaps)
      return low <= high ? continuous([low, high]) : null
    })
    this.spectrumMap = $derived(this.spectrumMaps ? this.spectrumMapAt(0) : null)
    this.tokens = $derived(this.#tokenMap.data ?? null)
    this.palette = $derived(this.tokens ? tokenColors(this.tokens) : null)
  }

  get fetching(): boolean {
    return this.#result.isFetching
  }

  get matches(): { galaxy: number; index: number }[] {
    return Array.from(this.galaxies.subarray(1), (galaxy, at) => ({
      galaxy,
      index: at + 1
    }))
  }

  readonly score = (value: number): string => `${SIMILARITY.short} ${value.toFixed(DECIMALS)}`

  readonly caption: Caption = (value) => `token ${value}`

  readonly maskColor = (value: number): RGB => (value ? [255, 255, 255] : [0, 0, 0])

  readonly maskTitle: Caption = (value) => (value ? 'masked' : 'unmasked')

  scoreAt(index: number): number {
    return this.#result.data?.scores[index] ?? 0
  }

  imageMapAt(index: number): Float32Array {
    return row(must(this.imageMaps, 'the image maps'), index, this.#patchCount)
  }

  spectrumMapAt(index: number): Float32Array | null {
    const maps = must(this.spectrumMaps, 'the spectrum maps')
    const map = row(maps, index, maps.length / this.galaxies.length)
    return Number.isNaN(map[0]) ? null : map
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
