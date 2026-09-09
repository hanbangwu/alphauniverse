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

export function row(values: Float32Array, index: number, patches: number): Float32Array {
  return values.subarray(index * patches, (index + 1) * patches)
}
