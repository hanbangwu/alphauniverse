import type { Extent } from './similarity'

export function extent(values: ArrayLike<number>): Extent {
  let low = Infinity
  let high = -Infinity
  for (let i = 0; i < values.length; i++) {
    if (values[i] < low) low = values[i]
    if (values[i] > high) high = values[i]
  }
  return [low, high]
}

export function mask(values: ArrayLike<number>, threshold: number, invert: boolean): Uint8Array {
  const out = new Uint8Array(values.length)
  for (let i = 0; i < out.length; i++) {
    out[i] = values[i] >= threshold === invert ? 0 : 1
  }
  return out
}

export function row(values: Float32Array, index: number, width: number): Float32Array {
  return values.subarray(index * width, (index + 1) * width)
}

export function matchMean(values: Float32Array, rows: number): Float32Array {
  const width = values.length / rows
  const mean = new Float32Array(width)
  for (let index = 1; index < rows; index++) {
    for (let slot = 0; slot < width; slot++) {
      mean[slot] += values[index * width + slot] / (rows - 1)
    }
  }
  return mean
}
