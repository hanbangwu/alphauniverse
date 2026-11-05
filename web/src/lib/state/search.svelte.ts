import type { CosineMode } from '$lib/client'
import { TOKEN_MODES, anywhereModes } from '$lib/data/modes'
import type { ModeName, ModeQuery, SaliencyQuery } from '$lib/data/similarity'
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
      anywhere: anywhereModes(
        TOKEN_MODES.filter((mode) => tokens[mode].length),
        imageAnywhere,
        spectrumAnywhere
      )
    }
  }

  modeSelection(
    galaxy: number | null,
    modes: ModeName[],
    imageAnywhere: boolean,
    spectrumAnywhere: boolean
  ): Omit<ModeQuery, 'matches'> | null {
    if (galaxy === null || !modes.length) return null
    return { galaxy, modes, anywhere: anywhereModes(modes, imageAnywhere, spectrumAnywhere) }
  }

  request<T extends object>(selection: T | null): (T & { matches: number }) | null {
    const matches = this.count
    return selection && matches !== null ? { ...selection, matches } : null
  }
}
