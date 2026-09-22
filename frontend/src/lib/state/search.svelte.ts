import type { SimilarityQuery } from '$lib/data/similarity'
import { NumberField } from './fields.svelte'
import { rangeOf } from './schema'

export class SearchState {
  readonly matches = new NumberField(rangeOf('matches'), 1, true)

  request(galaxy: number | null, patches: number[], spans: number[]): SimilarityQuery | null {
    if (galaxy === null || patches.length + spans.length === 0) return null
    return { galaxy, p: patches, s: spans, matches: this.matches.value }
  }
}
