import type { Extent } from '$lib/data/similarity'
import { Field } from './field.svelte'

export const DECIMALS = 3

export class Threshold {
  readonly override = new Field<number | null>(null)

  at([low, high]: Extent): number {
    const chosen = this.override.value
    return chosen === null ? (low + high) / 2 : Math.min(high, Math.max(low, chosen))
  }

  get pinned(): boolean {
    return this.override.value !== null
  }
}

export class MaskState {
  readonly on = new Field(false)
  readonly invert = new Field(false)
  readonly query = new Threshold()
  readonly matches = new Threshold()
}
