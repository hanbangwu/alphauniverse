import type { Meta } from '$lib/api'
import type { Extent } from './similarity'

export interface Spectrum {
  wavelength: Float32Array
  flux: Float32Array
}

export function spectrumTokenAt(meta: Meta, wavelength: number): number {
  return Math.floor((wavelength - meta.spectrum_origin) / meta.spectrum_width)
}

export function spectrumTokenOf(meta: Meta, index: number): Extent {
  return [
    meta.spectrum_origin + index * meta.spectrum_width,
    meta.spectrum_origin + (index + 1) * meta.spectrum_width
  ]
}
