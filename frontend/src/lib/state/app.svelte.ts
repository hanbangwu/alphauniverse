import type { Meta } from '$lib/api'
import { morphologyColors, propertyColors } from '$lib/color'
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
  readonly swatches: string[] | null
  readonly shades: string[] | null

  constructor(
    readonly meta: Meta,
    readonly mosaic: MosaicState
  ) {
    this.view = new ViewState()
    this.search = new SearchState()
    this.filters = new FilterState(meta, this.mosaic, this.view)
    this.swatches = $derived(
      mode.current ? morphologyColors(meta.morphologies.length - 1, mode.current) : null
    )
    this.shades = $derived(mode.current ? propertyColors(mode.current) : null)
  }

  get dataset(): string {
    return `${this.meta.author}/${this.meta.id}`
  }

  get shownPoints() {
    return pointsQuery(this.mosaic, this.view.pointSet.value)
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
