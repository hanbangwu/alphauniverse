import type { GetSpectrumData, SpectrumGrid, Survey } from '$lib/api'
import { zGetSpectrumPath } from '$lib/api/zod.gen'
import type { Extent } from './similarity'

export type SpectrumSurvey = GetSpectrumData['path']['survey']

export const SPECTRUM_SURVEYS: readonly SpectrumSurvey[] = zGetSpectrumPath.shape.survey.options

export interface Spectrum {
  wavelength: Float32Array
  flux: Float32Array
}

export function spectrumSurvey(coverage: Survey[]): SpectrumSurvey | null {
  return (
    SPECTRUM_SURVEYS.find((survey) =>
      coverage.some((row) => row.survey === survey && row.matched)
    ) ?? null
  )
}

export function spanAt(grid: SpectrumGrid, wavelength: number): number {
  return Math.floor((wavelength - grid.origin) / grid.width)
}

export function spanOf(grid: SpectrumGrid, index: number): Extent {
  return [grid.origin + index * grid.width, grid.origin + (index + 1) * grid.width]
}
