import type { SearchMethod } from '$lib/api'
import type { Extent, SimilarityResult } from './similarity'

export function extent(values: ArrayLike<number>): Extent {
  let low = Infinity
  let high = -Infinity
  for (let i = 0; i < values.length; i++) {
    if (values[i] < low) low = values[i]
    if (values[i] > high) high = values[i]
  }
  return [low, high]
}

export function blend(
  result: SimilarityResult,
  method: SearchMethod,
  weight: number
): Float32Array {
  switch (method) {
    case 'encoded':
      return result.encoded
    case 'codebook':
      return result.codebook
    case 'codebook_and_encoded':
      break
    default:
      throw new Error('search method')
  }

  const out = new Float32Array(result.encoded.length)
  for (let i = 0; i < out.length; i++) {
    out[i] = weight * result.codebook[i] + (1 - weight) * result.encoded[i]
  }
  return out
}

export function mask(values: ArrayLike<number>, threshold: number, invert: boolean): Uint8Array {
  const out = new Uint8Array(values.length)
  for (let i = 0; i < out.length; i++) {
    out[i] = values[i] >= threshold === invert ? 0 : 1
  }
  return out
}

export function row(values: Float32Array, index: number, patches: number): Float32Array {
  return values.subarray(index * patches, (index + 1) * patches)
}
