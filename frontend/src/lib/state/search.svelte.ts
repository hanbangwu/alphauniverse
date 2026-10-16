import type { CosineMode } from '$lib/client'
import {
  IMAGE_SURVEYS,
  SPECTRUM_SURVEYS,
  TOKEN_MODES,
  imageMode,
  spectrumMode
} from '$lib/data/modes'
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
    tokens: Record<CosineMode, number[]>,
    tableValues: number[],
    imageAnywhere: boolean,
    spectrumAnywhere: boolean
  ): SaliencyQuery | null {
    if (galaxy === null) return null
    if (TOKEN_MODES.every((mode) => !tokens[mode].length) && !tableValues.length) return null
    return {
      galaxy,
      ...tokens,
      table_values: tableValues,
      anywhere: [
        ...(imageAnywhere ? IMAGE_SURVEYS.map(imageMode) : []),
        ...(spectrumAnywhere ? SPECTRUM_SURVEYS.map(spectrumMode) : [])
      ].filter((mode) => tokens[mode].length)
    }
  }

  request(selection: SaliencyQuery | null): SimilarityQuery | null {
    const matches = this.count
    return selection && matches !== null ? { ...selection, matches } : null
  }
}
