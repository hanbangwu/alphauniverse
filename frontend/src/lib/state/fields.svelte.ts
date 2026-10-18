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

export class ListField<T> extends Field<T[]> {
  constructor() {
    super([])
  }

  has(entry: T): boolean {
    return this.value.includes(entry)
  }

  toggle(entry: T): void {
    this.value = this.has(entry)
      ? this.value.filter((other) => other !== entry)
      : [...this.value, entry]
  }
}

export class IndexListField extends ListField<number> {
  protected normalise(value: number[]): number[] {
    const sorted = value
      .filter((entry) => Number.isInteger(entry) && entry >= 0)
      .sort((a, b) => a - b)
    return sorted.filter((entry, index) => index === 0 || entry !== sorted[index - 1])
  }
}
