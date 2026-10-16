import type { CosineMode, ImageSurvey, SpectrumSurvey } from '$lib/client'

export const IMAGE_SURVEYS: readonly ImageSurvey[] = ['ls', 'hsc']
export const SPECTRUM_SURVEYS: readonly SpectrumSurvey[] = ['desi', 'sdss']

export const imageMode = (survey: ImageSurvey) => `${survey}_image` as const
export const spectrumMode = (survey: SpectrumSurvey) => `${survey}_spectrum` as const

export const TOKEN_MODES: readonly CosineMode[] = [
  ...IMAGE_SURVEYS.map(imageMode),
  ...SPECTRUM_SURVEYS.map(spectrumMode)
]

export function byMode<T>(make: (mode: CosineMode) => T): Record<CosineMode, T> {
  return Object.fromEntries(TOKEN_MODES.map((mode) => [mode, make(mode)])) as Record<CosineMode, T>
}
