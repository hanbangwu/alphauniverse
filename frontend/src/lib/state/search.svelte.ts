import type { SaliencyQuery, SimilarityQuery } from '$lib/data/similarity'
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

  selection(
    galaxy: number | null,
    imageTokens: number[],
    spectrumTokens: number[],
    tableValues: number[],
    imageAnywhere: boolean,
    spectrumAnywhere: boolean
  ): SaliencyQuery | null {
    if (galaxy === null) return null
    if (imageTokens.length + spectrumTokens.length + tableValues.length === 0) return null
    return {
      galaxy,
      ls_image: imageTokens,
      desi_spectrum: spectrumTokens,
      table_values: tableValues,
      anywhere: [
        ...(imageAnywhere && imageTokens.length ? (['ls_image'] as const) : []),
        ...(spectrumAnywhere && spectrumTokens.length ? (['desi_spectrum'] as const) : [])
      ]
    }
  }

  request(selection: SaliencyQuery | null): SimilarityQuery | null {
    const matches = this.count
    return selection && matches !== null ? { ...selection, matches } : null
  }
}
