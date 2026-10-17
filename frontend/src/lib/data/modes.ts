import type { CosineMode, ImageSurvey, SpectrumSurvey } from '$lib/client'

export const ANCHOR: ImageSurvey = 'ls'
export const IMAGE_SURVEYS: readonly ImageSurvey[] = [ANCHOR, 'hsc']
export const SPECTRUM_SURVEYS: readonly SpectrumSurvey[] = ['desi', 'sdss']

export const imageMode = (survey: ImageSurvey) => `${survey}_image` as const
export const spectrumMode = (survey: SpectrumSurvey) => `${survey}_spectrum` as const
export const surveyOf = (mode: CosineMode) => mode.split('_')[0] as ImageSurvey | SpectrumSurvey

export const IMAGE_MODES = IMAGE_SURVEYS.map(imageMode)
export const SPECTRUM_MODES = SPECTRUM_SURVEYS.map(spectrumMode)
export const TOKEN_MODES: readonly CosineMode[] = [...IMAGE_MODES, ...SPECTRUM_MODES]

export function byMode<T>(make: (mode: CosineMode) => T): Record<CosineMode, T> {
  return Object.fromEntries(TOKEN_MODES.map((mode) => [mode, make(mode)])) as Record<CosineMode, T>
}
