import type { CosineMode } from '$lib/client'
import { type RGB, continuous } from '$lib/color'
import { byMode } from '$lib/data/modes'
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

type Heat = (value: number) => RGB

export type Layer = 'galaxy' | 'population'

export class Similarity {
  readonly galaxy: number
  readonly grid: number

  readonly #app: AppState = getApp()
  readonly #result: CreateQueryResult<SimilarityResult>
  readonly #cosines: Record<CosineMode, CreateQueryResult<Float32Array>>
  readonly #saliency: CreateQueryResult<SaliencyResult>
  #submitted: SimilarityQuery | null = $state(null)
  readonly #selected: boolean

  layer: Layer = $state('galaxy')

  readonly draft: SimilarityQuery | null

  readonly galaxies: Int32Array
  readonly domains: Record<CosineMode, Extent> | null
  readonly heats: Record<CosineMode, Heat> | null
  readonly layers: Record<CosineMode, Float32Array | null>
  readonly layerDomains: Record<CosineMode, Extent | null>
  readonly layerHeats: Record<CosineMode, Heat | null>
  readonly tableValueMap: Float32Array | null
  readonly tableValueHeat: Heat | null

  constructor(galaxy: number) {
    const { view, search } = this.#app
    this.galaxy = galaxy
    this.grid = this.#app.meta.grid

    const selections = $derived(view.selections)
    const selection = $derived(
      search.selection(
        galaxy,
        selections,
        view.tableValues.value,
        view.imageAnywhere.value,
        view.spectrumAnywhere.value
      )
    )
    this.#selected = $derived(selection !== null)
    this.draft = $derived(search.request(selection))
    this.#result = createQuery(() => similarityQuery(this.#submitted))
    const own = $derived(this.layer === 'galaxy')
    this.#cosines = byMode((mode) =>
      createQuery(() => cosinesQuery(own && selections[mode].length ? galaxy : null, mode))
    )
    this.#saliency = createQuery(() => saliencyQuery(own ? null : selection))
    const saliency = $derived(own || !selection ? null : (this.#saliency.data ?? null))
    const result = $derived(this.#result.data)

    this.galaxies = $derived(result?.galaxies ?? new Int32Array())
    this.domains = $derived(
      result
        ? byMode((mode) => {
            const maps = result.maps[mode]
            return extent(maps.subarray(maps.length / result.galaxies.length))
          })
        : null
    )
    this.heats = $derived(this.domains && byMode((mode) => continuous(this.domains![mode])))
    this.layers = $derived(
      byMode((mode) => {
        if (saliency) return saliency.maps[mode]
        const cosines = this.#cosines[mode].data
        return cosines ? best(cosines, selections[mode]) : null
      })
    )
    this.layerDomains = $derived(
      byMode((mode) => {
        const layer = this.layers[mode]
        return layer && extent(layer)
      })
    )
    this.layerHeats = $derived(
      byMode((mode) => {
        const domain = this.layerDomains[mode]
        return domain && continuous(domain)
      })
    )
    this.tableValueMap = $derived(saliency ? saliency.tableValues : (result?.tableValueMap ?? null))
    this.tableValueHeat = $derived(
      this.tableValueMap ? continuous(extent(this.tableValueMap)) : null
    )
  }

  get error(): string | null {
    return this.#result.isError ? errorMessage(this.#result.error) : null
  }

  get saliencyError(): string | null {
    if (this.layer === 'galaxy') return null
    return this.#saliency.isError ? errorMessage(this.#saliency.error) : null
  }

  layerError(mode: CosineMode): string | null {
    if (this.layer !== 'galaxy') return null
    const cosines = this.#cosines[mode]
    return cosines.isError ? errorMessage(cosines.error) : null
  }

  anywhere(mode: CosineMode): boolean {
    return this.#result.data?.request.anywhere?.includes(mode) ?? false
  }

  outlinesAt(mode: CosineMode, index: number): number[] {
    const data = this.#result.data
    return data?.positions[mode][index] ?? data?.request[mode] ?? []
  }

  observes(survey: string, index: number): boolean {
    return this.#result.data?.observed[survey]?.[index] ?? true
  }

  get layerStale(): boolean {
    return this.layer !== 'galaxy' && this.#selected && this.#saliency.isPlaceholderData
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

  mapAt(mode: CosineMode, index: number): Float32Array {
    const maps = must(this.#result.data, 'the search result').maps[mode]
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
