import type { SpectrumGrid, Survey } from '$lib/api'
import type { Extent } from './similarity'

export const SPECTRUM_SURVEY = 'desi'

export interface Spectrum {
  wavelength: Float32Array
  flux: Float32Array
}

export function hasSpectrum(coverage: Survey[]): boolean {
  return coverage.some((row) => row.survey === SPECTRUM_SURVEY && row.matched)
}

export function spanAt(grid: SpectrumGrid, wavelength: number): number {
  return Math.floor((wavelength - grid.origin) / grid.width)
}

export function spanOf(grid: SpectrumGrid, index: number): Extent {
  return [grid.origin + index * grid.width, grid.origin + (index + 1) * grid.width]
}
