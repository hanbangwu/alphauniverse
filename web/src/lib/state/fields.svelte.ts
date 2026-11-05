import { must } from '$lib/invariant'
import type { Choice, Labels } from '$lib/labels'
import { Field, type Source } from './field.svelte'

export class EnumField<T extends string> extends Field<T> {
  readonly members: readonly T[]
  readonly choices: readonly Choice<T>[]

  constructor(
    private readonly labels: Labels<T>,
    source: Source<T>
  ) {
    const members = Object.keys(labels) as T[]
    super(must(members[0], 'a first member to fall back to'), source)
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
    return value
      .filter(
        (entry, index) => Number.isInteger(entry) && entry >= 0 && value.indexOf(entry) === index
      )
      .sort((a, b) => a - b)
  }
}
