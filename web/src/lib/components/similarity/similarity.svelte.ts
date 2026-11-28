import type { Galaxy } from '$lib/api'
import type { GalaxyId, TokenMode } from '$lib/client'
import { type RGB, continuous } from '$lib/color'
import { IMAGE_MODES, SPECTRUM_MODES, byMode, isPatchMode, surveyOf } from '$lib/data/modes'
import { coverageQuery, saliencyQuery, similarityQuery } from '$lib/data/queries'
import { extent, mask, row } from '$lib/data/scores'
import {
  type Extent,
  type ModeName,
  type SaliencyResult,
  type SearchQuery,
  type SimilarityResult,
  isModeQuery
} from '$lib/data/similarity'
import { must } from '$lib/invariant'
import { MODES } from '$lib/labels'
import { type AppState, getApp } from '$lib/state/app.svelte'
import { Field } from '$lib/state/field.svelte'
import { DECIMALS, type Threshold } from '$lib/state/mask.svelte'
import { DEFAULTS, stored } from '$lib/state/params.svelte'
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

function scale(domain: Extent): Scale {
  return { domain, heat: continuous(domain) }
}

function union(domains: (Extent | undefined)[]): Extent | null {
  const ends = domains.flatMap((domain) => domain ?? [])
  return ends.length ? extent(ends) : null
}

export class Similarity {
  readonly galaxy: GalaxyId

  readonly #app: AppState = getApp()
  readonly #result: CreateQueryResult<SimilarityResult>
  readonly #coverage: CreateQueryResult<Galaxy>
  readonly #layers: Record<TokenMode, Layered | null>
  readonly #saliency: CreateQueryResult<SaliencyResult>
  readonly #selected: boolean

  readonly imageFalseColor: Field<boolean>
  readonly spectrumFalseColor: Field<boolean>

  readonly draft: SearchQuery | null

  readonly galaxies: GalaxyId[]
  readonly scales: Record<TokenMode, Scale | null> | null
  readonly imageDomain: Extent | null
  readonly spectrumDomain: Extent | null
  readonly tableValueMap: Float32Array | null
  readonly tableValueDomain: Extent | null
  readonly tableValueHeat: Heat | null
  readonly contributionModes: ModeName[]
  readonly contributionLimit: number
  readonly modes: ModeName[]

  constructor(galaxy: GalaxyId) {
    const { view, search } = this.#app
    this.galaxy = galaxy
    this.imageFalseColor = new Field(DEFAULTS.image_false_color, stored('image_false_color'))
    this.spectrumFalseColor = new Field(
      DEFAULTS.spectrum_false_color,
      stored('spectrum_false_color')
    )

    const whole = $derived(view.searchBy.value === 'whole')
    const selections = $derived(byMode((mode) => view.patches[mode].value))
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
    this.modes = $derived(
      (Object.keys(MODES) as ModeName[]).filter(
        (mode) => view.modes.has(mode) && (!isPatchMode(mode) || this.shown(mode))
      )
    )
    this.draft = $derived(
      search.request(
        whole
          ? search.modeSelection(
              galaxy,
              this.modes,
              view.imageAnywhere.value,
              view.spectrumAnywhere.value
            )
          : selection
      )
    )
    this.#result = createQuery(() => similarityQuery(view.searched.value))
    this.#coverage = createQuery(() => coverageQuery(galaxy))
    this.#saliency = createQuery(() => saliencyQuery(selection))
    const saliency = $derived(selection ? (this.#saliency.data ?? null) : null)
    this.#layers = $derived(
      byMode((mode) => {
        const values = saliency?.maps[mode]
        return values ? { values, ...scale(extent(values)) } : null
      })
    )
    const result = $derived(this.#result.data)

    this.galaxies = $derived(result?.galaxies ?? [])
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

  patchesOf(mode: TokenMode): number[] {
    const { view } = this.#app
    return view.searchBy.value === 'whole' ? [] : view.patches[mode].value
  }

  toggler(mode: TokenMode): ((index: number) => void) | undefined {
    const { view } = this.#app
    return view.searchBy.value === 'patch' ? (index) => view.patches[mode].toggle(index) : undefined
  }

  get result(): SimilarityResult | undefined {
    return this.#result.data
  }

  get error(): string | null {
    return this.#result.isError ? this.#result.error.message : null
  }

  get saliencyError(): string | null {
    return this.#saliency.isError ? this.#saliency.error.message : null
  }

  layerOf(mode: TokenMode): Layered | null {
    return this.#layers[mode]
  }

  get imageLayerDomain(): Extent | null {
    return this.#layerDomain(IMAGE_MODES)
  }

  get spectrumLayerDomain(): Extent | null {
    return this.#layerDomain(SPECTRUM_MODES)
  }

  shown(mode: TokenMode): boolean {
    const survey = surveyOf(mode)
    return (
      this.#app.view.dataSource.value === 'prediction' || (this.#coverage.data?.[survey] ?? false)
    )
  }

  #layerDomain(modes: readonly TokenMode[]): Extent | null {
    return union(modes.filter((mode) => this.shown(mode)).map((mode) => this.layerOf(mode)?.domain))
  }

  outlinesAt(mode: TokenMode, index: number): number[] {
    const data = this.#result.data
    if (!data || isModeQuery(data.request)) return []
    return data.positions[mode]?.[index] ?? data.request[mode] ?? []
  }

  observes(survey: string, index: number): boolean {
    return this.#result.data?.observed[survey]?.[index] ?? false
  }

  shows(survey: string, index: number): boolean {
    return this.#app.view.dataSource.value === 'prediction' || this.observes(survey, index)
  }

  get layerStale(): boolean {
    return this.#selected && this.#saliency.isPlaceholderData
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

  get matches(): { galaxy: GalaxyId; index: number }[] {
    return this.galaxies.slice(1).map((galaxy, at) => ({
      galaxy,
      index: at + 1
    }))
  }

  get wholeResult(): boolean {
    const request = this.#result.data?.request
    return request !== undefined && isModeQuery(request)
  }

  readonly score = (value: number): string => `similarity ${Math.min(1, value).toFixed(DECIMALS)}`

  readonly total = (value: number): string =>
    this.wholeResult ? `score ${value.toFixed(DECIMALS)}` : this.score(value)

  readonly correlation: Caption = (value) => `partial correlation ${value.toFixed(DECIMALS)}`

  predictedAt(index: number): string[] {
    return this.#result.data?.predicted[index] ?? []
  }

  scoreAt(index: number): number {
    return this.#result.data?.scores[index] ?? 0
  }

  contributionAt(mode: ModeName, index: number): number {
    return must(this.#result.data?.contributions[mode], `${mode}'s contributions`)[index]
  }

  mapAt(mode: TokenMode, index: number): Float32Array {
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
