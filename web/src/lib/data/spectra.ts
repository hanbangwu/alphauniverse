import type { Extent } from './similarity'

const SPECTRUM_ORIGIN = 3500
const SPECTRUM_PATCH_WIDTH = 25.6
export const SPECTRUM_PATCHES = 272

export interface Spectrum {
  wavelength: Float32Array
  flux: Float32Array
}

export function spectrumPatchAt(wavelength: number): number {
  return Math.floor((wavelength - SPECTRUM_ORIGIN) / SPECTRUM_PATCH_WIDTH)
}

export function spectrumPatchOf(index: number): Extent {
  return [
    SPECTRUM_ORIGIN + index * SPECTRUM_PATCH_WIDTH,
    SPECTRUM_ORIGIN + (index + 1) * SPECTRUM_PATCH_WIDTH
  ]
}
