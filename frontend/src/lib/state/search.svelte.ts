import type { SimilarityQuery } from '$lib/data/similarity'
import { Field } from './field.svelte'
import { rangeOf } from './schema'

export class SearchState {
  readonly range = rangeOf('matches')
  readonly matches = new Field(String(this.range.default))

  get count(): number | null {
    const value = Number(this.matches.value)
    const { minimum, maximum } = this.range
    return Number.isInteger(value) && value >= minimum && value <= maximum ? value : null
  }

  request(galaxy: number | null, patches: number[], spans: number[]): SimilarityQuery | null {
    const matches = this.count
    if (galaxy === null || matches === null || patches.length + spans.length === 0) return null
    return { galaxy, p: patches, s: spans, matches }
  }
}
