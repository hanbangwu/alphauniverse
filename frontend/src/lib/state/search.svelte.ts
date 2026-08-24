import type { CombineMode, SearchMethod } from '$lib/api'
import type { SimilarityQuery } from '$lib/data/similarity'
import { COMBINES, METHODS } from '$lib/labels'
import { EnumField, NumberField } from './fields.svelte'
import { optionsOf, rangeOf } from './schema'

export class SearchState {
  readonly method: EnumField<SearchMethod>
  readonly combine: EnumField<CombineMode>
  readonly weight = new NumberField(rangeOf('weight'), 0.01)
  readonly neighbours = new NumberField(rangeOf('neighbours'), 1, true)
  readonly matches = new NumberField(rangeOf('matches'), 1, true)

  constructor() {
    this.method = new EnumField(optionsOf<SearchMethod>('method'), METHODS)
    this.combine = new EnumField(optionsOf<CombineMode>('combine'), COMBINES)
  }

  get blends(): boolean {
    return this.method.value === 'codebook_and_encoded'
  }

  request(galaxy: number | null, patches: number[]): SimilarityQuery | null {
    if (galaxy === null || patches.length === 0) return null
    return {
      galaxy,
      p: patches,
      method: this.method.value,
      combine: this.combine.value,
      neighbours: this.neighbours.value,
      matches: this.matches.value,
      ...(this.blends ? { weight: this.weight.value } : {})
    }
  }
}
