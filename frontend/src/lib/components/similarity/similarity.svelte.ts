import type { Galaxy } from '$lib/api'
import type { CosineMode } from '$lib/client'
import { type RGB, continuous } from '$lib/color'
import { ANCHOR, IMAGE_MODES, SPECTRUM_MODES, byMode, surveyOf } from '$lib/data/modes'
import { cosinesQuery, coverageQuery, saliencyQuery, similarityQuery } from '$lib/data/queries'
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

interface Scale {
  domain: Extent
  heat: Heat
}

interface Layered extends Scale {
  values: Float32Array
}

export type Layer = 'galaxy' | 'population'

function scale(domain: Extent): Scale {
  return { domain, heat: continuous(domain) }
}

function union(domains: (Extent | undefined)[]): Extent | null {
  const ends = domains.flatMap((domain) => domain ?? [])
  return ends.length ? extent(ends) : null
}

class ModeLayer {
  readonly #cosines: CreateQueryResult<Float32Array>
  readonly #own: () => boolean
  readonly current: Layered | null

  constructor(
    mode: CosineMode,
    galaxy: number,
    selected: () => number[],
    own: () => boolean,
    saliency: () => SaliencyResult | null
  ) {
    this.#own = own
    this.#cosines = createQuery(() =>
      cosinesQuery(own() && selected().length ? galaxy : null, mode)
    )
    this.current = $derived.by(() => {
      const population = saliency()
      const cosines = this.#cosines.data
      const values = population ? population.maps[mode] : cosines ? best(cosines, selected()) : null
      return values && { values, ...scale(extent(values)) }
    })
  }

  get error(): string | null {
    if (!this.#own()) return null
    return this.#cosines.isError ? errorMessage(this.#cosines.error) : null
  }
}

export class Similarity {
  readonly galaxy: number
  readonly grid: number

  readonly #app: AppState = getApp()
  readonly #result: CreateQueryResult<SimilarityResult>
  readonly #coverage: CreateQueryResult<Galaxy>
  readonly #layers: Record<CosineMode, ModeLayer>
  readonly #saliency: CreateQueryResult<SaliencyResult>
  readonly #selected: boolean

  layer: Layer = $state('galaxy')

  readonly draft: SimilarityQuery | null

  readonly galaxies: Int32Array
  readonly scales: Record<CosineMode, Scale | null> | null
  readonly imageDomain: Extent | null
  readonly spectrumDomain: Extent | null
  readonly tableValueMap: Float32Array | null
  readonly tableValueDomain: Extent | null
  readonly tableValueHeat: Heat | null

  constructor(galaxy: number) {
    const { view, search } = this.#app
    this.galaxy = galaxy
    this.grid = this.#app.meta.grid

    const selections = $derived(byMode((mode) => view.tokens[mode].value))
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
    this.#result = createQuery(() => similarityQuery(view.searched.value))
    this.#coverage = createQuery(() => coverageQuery(galaxy))
    const own = $derived(this.layer === 'galaxy')
    this.#saliency = createQuery(() => saliencyQuery(own ? null : selection))
    const saliency = $derived(own || !selection ? null : (this.#saliency.data ?? null))
    this.#layers = byMode(
      (mode) =>
        new ModeLayer(
          mode,
          galaxy,
          () => view.tokens[mode].value,
          () => own,
          () => saliency
        )
    )
    const result = $derived(this.#result.data)

    this.galaxies = $derived(result?.galaxies ?? new Int32Array())
    this.scales = $derived(
      result
        ? byMode((mode) => {
            const maps = result.maps[mode]
            const width = maps.length / result.galaxies.length
            const domain = union(
              this.matches
                .filter(({ index }) => this.observes(surveyOf(mode), index))
                .map(({ index }) => extent(row(maps, index, width)))
            )
            return domain && scale(domain)
          })
        : null
    )
    this.imageDomain = $derived(union(IMAGE_MODES.map((mode) => this.scales?.[mode]?.domain)))
    this.spectrumDomain = $derived(union(SPECTRUM_MODES.map((mode) => this.scales?.[mode]?.domain)))
    this.tableValueMap = $derived(saliency ? saliency.tableValues : (result?.tableValueMap ?? null))
    this.tableValueDomain = $derived(this.tableValueMap && extent(this.tableValueMap))
    this.tableValueHeat = $derived(this.tableValueDomain && continuous(this.tableValueDomain))
  }

  get error(): string | null {
    return this.#result.isError ? errorMessage(this.#result.error) : null
  }

  get saliencyError(): string | null {
    if (this.layer === 'galaxy') return null
    return this.#saliency.isError ? errorMessage(this.#saliency.error) : null
  }

  layerOf(mode: CosineMode): Layered | null {
    return this.#layers[mode].current
  }

  get imageLayerDomain(): Extent | null {
    return this.#layerDomain(IMAGE_MODES)
  }

  get spectrumLayerDomain(): Extent | null {
    return this.#layerDomain(SPECTRUM_MODES)
  }

  #layerDomain(modes: readonly CosineMode[]): Extent | null {
    const coverage = this.#coverage.data
    return union(
      modes
        .filter((mode) => surveyOf(mode) === ANCHOR || coverage?.[surveyOf(mode)])
        .map((mode) => this.layerOf(mode)?.domain)
    )
  }

  layerError(mode: CosineMode): string | null {
    return this.#layers[mode].error
  }

  anywhere(mode: CosineMode): boolean {
    return this.#result.data?.request.anywhere?.includes(mode) ?? false
  }

  outlinesAt(mode: CosineMode, index: number): number[] {
    const data = this.#result.data
    return data?.positions[mode][index] ?? data?.request[mode] ?? []
  }

  observes(survey: string, index: number): boolean {
    return survey === ANCHOR || (this.#result.data?.observed[survey]?.[index] ?? false)
  }

  get layerStale(): boolean {
    return this.layer !== 'galaxy' && this.#selected && this.#saliency.isPlaceholderData
  }

  get stale(): boolean {
    return this.#result.isPlaceholderData
  }

  get searched(): boolean {
    return this.#app.view.searched.value !== null
  }

  submit(): void {
    this.#app.view.searched.value = this.draft
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
