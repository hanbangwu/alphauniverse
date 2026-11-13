import type { Extent } from './similarity'

const SPECTRUM_ORIGIN = 3500
const SPECTRUM_TOKEN_WIDTH = 25.6

export interface Spectrum {
  wavelength: Float32Array
  flux: Float32Array
}

export function spectrumTokenAt(wavelength: number): number {
  return Math.floor((wavelength - SPECTRUM_ORIGIN) / SPECTRUM_TOKEN_WIDTH)
}

export function spectrumTokenOf(index: number): Extent {
  return [
    SPECTRUM_ORIGIN + index * SPECTRUM_TOKEN_WIDTH,
    SPECTRUM_ORIGIN + (index + 1) * SPECTRUM_TOKEN_WIDTH
  ]
}
