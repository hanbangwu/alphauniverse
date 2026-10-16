import { must } from '$lib/invariant'
import type { Choice, Labels } from '$lib/labels'
import { Field } from './field.svelte'

export class EnumField<T extends string> extends Field<T> {
  readonly members: readonly T[]
  readonly choices: readonly Choice<T>[]

  constructor(private readonly labels: Labels<T>) {
    const members = Object.keys(labels) as T[]
    super(must(members[0], 'a first member to fall back to'))
    this.members = members
    this.choices = members.map((value) => ({ value, ...labels[value] }))
  }

  get label(): string {
    return this.labels[this.value].label
  }
}

export class IndexListField extends Field<number[]> {
  constructor() {
    super([])
  }

  protected normalise(value: number[]): number[] {
    const sorted = value
      .filter((entry) => Number.isInteger(entry) && entry >= 0)
      .sort((a, b) => a - b)
    return sorted.filter((entry, index) => index === 0 || entry !== sorted[index - 1])
  }

  has(index: number): boolean {
    return this.value.includes(index)
  }

  toggle(index: number): void {
    this.value = this.has(index)
      ? this.value.filter((entry) => entry !== index)
      : [...this.value, index]
  }
}
