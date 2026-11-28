import type { TableRow } from '$lib/api'
import type { TokenMode, ImageSurvey, SpectrumSurvey } from '$lib/client'
import { surveysOf } from '$lib/state/schema'
import type { ModeName } from './similarity'

export const IMAGE_GRID = 24

export const IMAGE_SURVEYS = surveysOf('/galaxy/{galaxy}/image/{survey}') as readonly ImageSurvey[]
export const SPECTRUM_SURVEYS = surveysOf(
  '/galaxy/{galaxy}/spectrum/{survey}'
) as readonly SpectrumSurvey[]

export const imageMode = (survey: ImageSurvey) => `${survey}_image` as const
export const spectrumMode = (survey: SpectrumSurvey) => `${survey}_spectrum` as const
export const surveyOf = (mode: TokenMode) => mode.split('_')[0] as ImageSurvey | SpectrumSurvey

export const IMAGE_MODES = IMAGE_SURVEYS.map(imageMode)
export const SPECTRUM_MODES = SPECTRUM_SURVEYS.map(spectrumMode)
export const PATCH_MODES: readonly TokenMode[] = [...IMAGE_MODES, ...SPECTRUM_MODES]

export function byMode<T>(make: (mode: TokenMode) => T): Record<TokenMode, T> {
  return Object.fromEntries(PATCH_MODES.map((mode) => [mode, make(mode)])) as Record<TokenMode, T>
}

export const TABLE_MODES: Partial<Record<TableRow['section'], ModeName>> = {
  redshift: 'redshift',
  ls: 'ls_table',
  hsc: 'hsc_table'
}

export const isPatchMode = (mode: ModeName): mode is TokenMode =>
  (PATCH_MODES as readonly ModeName[]).includes(mode)

export function anywhereModes(
  modes: readonly ModeName[],
  imageAnywhere: boolean,
  spectrumAnywhere: boolean
): TokenMode[] {
  return [
    ...(imageAnywhere ? IMAGE_MODES : []),
    ...(spectrumAnywhere ? SPECTRUM_MODES : [])
  ].filter((mode) => modes.includes(mode))
}
