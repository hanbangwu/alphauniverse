import type { Extent } from '$lib/data/similarity'
import { Field, type Source } from './field.svelte'
import { DEFAULTS, stored } from './params.svelte'

export const DECIMALS = 3

export class Threshold {
  readonly override: Field<number | null>
  dragged = $state<number | null>(null)

  constructor(source: Source<number | null>) {
    this.override = new Field(null, source)
  }

  at([low, high]: Extent): number {
    const chosen = this.dragged ?? this.override.value
    return chosen === null ? (low + high) / 2 : Math.min(high, Math.max(low, chosen))
  }

  get pinned(): boolean {
    return this.override.value !== null
  }
}

export class MaskState {
  readonly on: Field<boolean>
  readonly invert: Field<boolean>
  readonly images: { query: Threshold; matches: Threshold }
  readonly spectra: { query: Threshold; matches: Threshold }
  readonly table: Threshold

  constructor() {
    this.on = new Field(DEFAULTS.mask, stored('mask'))
    this.invert = new Field(DEFAULTS.invert, stored('invert'))
    this.images = {
      query: new Threshold(stored('image_threshold')),
      matches: new Threshold(stored('image_match_threshold'))
    }
    this.spectra = {
      query: new Threshold(stored('spectrum_threshold')),
      matches: new Threshold(stored('spectrum_match_threshold'))
    }
    this.table = new Threshold(stored('table_threshold'))
  }
}
