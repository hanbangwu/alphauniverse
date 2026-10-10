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

  request(
    galaxy: number | null,
    imageTokens: number[],
    spectrumTokens: number[],
    tableValues: number[]
  ): SimilarityQuery | null {
    const matches = this.count
    if (galaxy === null || matches === null) return null
    if (imageTokens.length + spectrumTokens.length + tableValues.length === 0) return null
    return { galaxy, p: imageTokens, s: spectrumTokens, t: tableValues, matches }
  }
}
