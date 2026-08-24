import type { Meta } from '$lib/api'
import { morphologyColors } from '$lib/color'
import { type PointsQuery, pointsQuery } from '$lib/data/queries'
import { checkMorphologies } from '$lib/labels'
import { FilterState } from './filter.svelte'
import { MaskState } from './mask.svelte'
import { MosaicState } from './mosaic.svelte'
import { SearchState } from './search.svelte'
import { ViewState } from './view.svelte'
import { mode } from 'mode-watcher'
import { Context } from 'runed'
import { onDestroy } from 'svelte'

export class AppState {
  readonly view: ViewState
  readonly search: SearchState
  readonly mask = new MaskState()
  readonly filters: FilterState
  readonly mosaic = new MosaicState()

  constructor(readonly meta: Meta) {
    checkMorphologies(meta.morphologies.length)
    this.view = new ViewState(meta)
    this.search = new SearchState()
    this.filters = new FilterState(meta, this.mosaic)
  }

  get dataset(): string {
    return `${this.meta.author}/${this.meta.id}`
  }

  get swatches(): string[] | null {
    return mode.current ? morphologyColors(this.meta.morphologies.length, mode.current) : null
  }

  get meanPoints(): PointsQuery {
    return pointsQuery(this.mosaic, this.meta, this.meta.mean_points)
  }

  get fullPoints(): PointsQuery {
    return pointsQuery(this.mosaic, this.meta, this.meta.full_points, this.view.full)
  }

  start(): void {
    this.filters.start()
    onDestroy(() => this.mosaic.destroy())
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
