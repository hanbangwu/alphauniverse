import type { Meta } from '$lib/api'
import { FilterState } from './filter.svelte'
import { MaskState } from './mask.svelte'
import type { MosaicState } from './mosaic.svelte'
import { SearchState } from './search.svelte'
import { ViewState } from './view.svelte'
import { Context } from 'runed'

export class AppState {
  readonly view: ViewState
  readonly search: SearchState
  readonly mask: MaskState
  readonly filters: FilterState

  constructor(
    readonly meta: Meta,
    readonly mosaic: MosaicState
  ) {
    this.view = new ViewState()
    this.search = new SearchState()
    this.mask = new MaskState()
    this.filters = new FilterState(this.mosaic, this.view)
    const searched = this.view.searched.value
    if (searched) {
      this.view.edit(searched)
      if (searched.matches) this.search.matches.value = String(searched.matches)
    }
  }

  get dataset(): string {
    return `${this.meta.author}/${this.meta.id}`
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
