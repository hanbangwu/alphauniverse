import type { SimilarityQuery } from '$lib/data/similarity'
import { NumberField } from './fields.svelte'
import { rangeOf } from './schema'

export class SearchState {
  readonly matches = new NumberField(rangeOf('matches'), 1, true)

  request(galaxy: number | null, patches: number[]): SimilarityQuery | null {
    if (galaxy === null || patches.length === 0) return null
    return { galaxy, p: patches, matches: this.matches.value }
  }
}
