import type { Galaxy, Meta } from '$lib/api'
import type { Extent } from './similarity'

export const SPECTRUM_SURVEY = 'desi'

export interface Spectrum {
  wavelength: Float32Array
  flux: Float32Array
}

export function hasSpectrum(galaxy: Galaxy): boolean {
  return galaxy[SPECTRUM_SURVEY]
}

export function spanAt(meta: Meta, wavelength: number): number {
  return Math.floor((wavelength - meta.spectrum_origin) / meta.spectrum_width)
}

export function spanOf(meta: Meta, index: number): Extent {
  return [
    meta.spectrum_origin + index * meta.spectrum_width,
    meta.spectrum_origin + (index + 1) * meta.spectrum_width
  ]
}
