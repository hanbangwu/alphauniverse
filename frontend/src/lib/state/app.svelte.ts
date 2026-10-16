import type { Meta } from '$lib/api'
import { morphologyColors } from '$lib/color'
import { pointsQuery } from '$lib/data/queries'
import { FilterState } from './filter.svelte'
import { MaskState } from './mask.svelte'
import type { MosaicState } from './mosaic.svelte'
import { SearchState } from './search.svelte'
import { ViewState } from './view.svelte'
import { mode } from 'mode-watcher'
import { Context } from 'runed'

export class AppState {
  readonly view: ViewState
  readonly search: SearchState
  readonly mask = new MaskState()
  readonly filters: FilterState

  constructor(
    readonly meta: Meta,
    readonly mosaic: MosaicState
  ) {
    this.view = new ViewState()
    this.search = new SearchState()
    this.filters = new FilterState(meta, this.mosaic)
  }

  get dataset(): string {
    return `${this.meta.author}/${this.meta.id}`
  }

  get swatches(): string[] | null {
    return mode.current ? morphologyColors(this.meta.morphologies.length - 1, mode.current) : null
  }

  get meanPoints() {
    return pointsQuery(this.mosaic, 'mean')
  }

  get fullPoints() {
    return pointsQuery(this.mosaic, 'full', this.view.full)
  }

  start(): void {
    this.filters.start()
  }
}

const context = new Context<AppState>('alphauniverse')

export const setApp = (app: AppState): AppState => context.set(app)
export const getApp = (): AppState => context.get()
export const getMeta = (): Meta => context.get().meta
export const getView = (): ViewState => context.get().view
export const getSearch = (): SearchState => context.get().search
export const getMask = (): MaskState => context.get().mask
export const getFilters = (): FilterState => context.get().filters
