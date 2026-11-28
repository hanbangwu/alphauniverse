import type { GalaxyId, TokenMode } from '$lib/client'
import { PATCH_MODES, anywhereModes } from '$lib/data/modes'
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
    galaxy: GalaxyId | null,
    patches: Record<TokenMode, number[]>,
    tableValues: number[],
    imageAnywhere: boolean,
    spectrumAnywhere: boolean
  ): SaliencyQuery | null {
    if (galaxy === null) return null
    if (PATCH_MODES.every((mode) => !patches[mode].length) && !tableValues.length) return null
    return {
      galaxy,
      ...patches,
      table_values: tableValues,
      anywhere: anywhereModes(
        PATCH_MODES.filter((mode) => patches[mode].length),
        imageAnywhere,
        spectrumAnywhere
      )
    }
  }

  modeSelection(
    galaxy: GalaxyId | null,
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
