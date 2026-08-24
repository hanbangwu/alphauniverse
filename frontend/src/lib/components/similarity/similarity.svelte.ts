import { type RGB, continuous, tokenColors } from '$lib/color'
import { similarityQuery, tokensQuery } from '$lib/data/queries'
import { blend, extent, mask, row } from '$lib/data/scores'
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

  readonly blended: Float32Array | null
  readonly galaxies: Int32Array
  readonly domain: Extent | null
  readonly heat: ((value: number) => RGB) | null
  readonly self: Float32Array | null
  readonly palette: ((value: number) => RGB) | null
  readonly cells: ArrayLike<number> | null

  constructor(galaxy: number) {
    const app = this.#app
    this.galaxy = galaxy
    this.grid = app.meta.grid

    this.#tokenMap = createQuery(() => tokensQuery(app.meta, galaxy))
    this.#result = createQuery(() =>
      similarityQuery(app.meta, app.search.request(galaxy, app.view.patches.value))
    )

    const tokens = $derived(this.#tokenMap.data ?? null)

    this.blended = $derived(
      this.#result.data
        ? blend(this.#result.data, app.search.method.value, app.search.weight.value)
        : null
    )
    this.galaxies = $derived(this.#result.data?.galaxies ?? new Int32Array())
    this.domain = $derived(this.blended ? extent(this.blended) : null)
    this.heat = $derived(this.domain ? continuous(this.domain) : null)
    this.self = $derived(this.blended ? row(this.blended, 0, this.#patchCount) : null)
    this.palette = $derived(
      this.self && this.heat ? this.heat : tokens ? tokenColors(tokens) : null
    )
    this.cells = $derived(this.self ?? tokens)
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

  readonly score: Caption = (value) => `${SIMILARITY.short} ${value.toFixed(DECIMALS)}`

  readonly caption: Caption = (value, index) =>
    this.self ? this.score(value, index) : `token ${value}`

  readonly maskColor = (value: number): RGB => (value ? [255, 255, 255] : [0, 0, 0])

  readonly maskTitle: Caption = (value) => (value ? 'masked' : 'unmasked')

  scoreAt(index: number): number {
    return this.#result.data?.scores[index] ?? 0
  }

  rowAt(index: number): Float32Array {
    return row(must(this.blended, 'the blended scores'), index, this.#patchCount)
  }

  maskOf(values: ArrayLike<number>): Uint8Array | null {
    const display = this.#app.mask
    if (!display.on.value || !this.domain) return null
    return mask(values, display.at(this.domain), display.invert.value)
  }
}

const context = new Context<Similarity>('similarity')

export const setSimilarity = (value: Similarity): Similarity => context.set(value)
export const getSimilarity = (): Similarity => context.get()
