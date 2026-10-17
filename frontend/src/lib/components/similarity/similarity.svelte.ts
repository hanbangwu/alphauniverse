import type { Galaxy } from '$lib/api'
import type { CosineMode } from '$lib/client'
import { type RGB, continuous } from '$lib/color'
import { ANCHOR, IMAGE_MODES, SPECTRUM_MODES, byMode, surveyOf } from '$lib/data/modes'
import { cosinesQuery, coverageQuery, saliencyQuery, similarityQuery } from '$lib/data/queries'
import { best, extent, mask, row } from '$lib/data/scores'
import {
  type Extent,
  type ModeName,
  type SaliencyResult,
  type SearchQuery,
  type SimilarityResult,
  isModeQuery
} from '$lib/data/similarity'
import { errorMessage } from '$lib/errors'
import { must } from '$lib/invariant'
import { MODES, OBSERVATIONS, SIMILARITY } from '$lib/labels'
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

  readonly draft: SearchQuery | null

  readonly galaxies: Int32Array
  readonly scales: Record<CosineMode, Scale | null> | null
  readonly imageDomain: Extent | null
  readonly spectrumDomain: Extent | null
  readonly tableValueMap: Float32Array | null
  readonly tableValueDomain: Extent | null
  readonly tableValueHeat: Heat | null
  readonly contributionModes: ModeName[]
  readonly contributionLimit: number

  constructor(galaxy: number) {
    const { view, search } = this.#app
    this.galaxy = galaxy
    this.grid = this.#app.meta.grid

    const whole = $derived(view.wholeModes.value)
    const selections = $derived(byMode((mode) => this.tokensOf(mode)))
    const selection = $derived(
      whole
        ? null
        : search.selection(
            galaxy,
            selections,
            view.tableValues.value,
            view.imageAnywhere.value,
            view.spectrumAnywhere.value
          )
    )
    this.#selected = $derived(selection !== null)
    this.draft = $derived(
      search.request(
        whole
          ? search.modeSelection(
              galaxy,
              view.modes.value,
              view.imageAnywhere.value,
              view.spectrumAnywhere.value
            )
          : selection
      )
    )
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
          () => this.tokensOf(mode),
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
                .filter(({ index }) => this.shows(surveyOf(mode), index))
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
    const contributions = $derived(result?.contributions ?? {})
    this.contributionModes = $derived(
      (Object.keys(MODES) as ModeName[]).filter((mode) => mode in contributions)
    )
    this.contributionLimit = $derived(
      Object.values(contributions).reduce(
        (limit, values) =>
          values.subarray(1).reduce((most, value) => Math.max(most, Math.abs(value)), limit),
        0
      )
    )
  }

  tokensOf(mode: CosineMode): number[] {
    const { view } = this.#app
    return view.wholeModes.value ? [] : view.tokens[mode].value
  }

  toggler(mode: CosineMode, matched: boolean): ((index: number) => void) | undefined {
    const { view } = this.#app
    return matched && !view.wholeModes.value
      ? (index) => view.tokens[mode].toggle(index)
      : undefined
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
        .filter(
          (mode) =>
            this.#app.view.predictions.value ||
            surveyOf(mode) === ANCHOR ||
            coverage?.[surveyOf(mode)]
        )
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
    if (!data || isModeQuery(data.request)) return []
    return data.positions[mode]?.[index] ?? data.request[mode] ?? []
  }

  observes(survey: string, index: number): boolean {
    return survey === ANCHOR || (this.#result.data?.observed[survey]?.[index] ?? false)
  }

  shows(survey: string, index: number): boolean {
    return this.#app.view.predictions.value || this.observes(survey, index)
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

  get wholeModes(): boolean {
    const request = this.#result.data?.request
    return request !== undefined && isModeQuery(request)
  }

  readonly score = (value: number): string =>
    `${SIMILARITY.short} ${Math.min(1, value).toFixed(DECIMALS)}`

  readonly total = (value: number): string =>
    this.wholeModes ? `score ${value.toFixed(DECIMALS)}` : this.score(value)

  readonly cosine: Caption = (value) => `cosine ${value.toFixed(DECIMALS)}`

  readonly correlation: Caption = (value) => `partial correlation ${value.toFixed(DECIMALS)}`

  get layerCaption(): Caption {
    return this.layer === 'galaxy' ? this.cosine : this.correlation
  }

  readonly caption: Caption = (value) => `token ${value}`

  predictedAt(index: number): string[] {
    return (this.#result.data?.predicted[index] ?? []).map((column) => OBSERVATIONS[column])
  }

  scoreAt(index: number): number {
    return this.#result.data?.scores[index] ?? 0
  }

  contributionAt(mode: ModeName, index: number): number {
    return must(this.#result.data?.contributions[mode], `${mode}'s contributions`)[index]
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
