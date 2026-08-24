import type { Choice, Labels } from '$lib/labels'
import { Field } from './field.svelte'
import type { Options, Range } from './schema'

export class EnumField<T extends string> extends Field<T> {
  readonly members: readonly T[]
  readonly choices: readonly Choice<T>[]

  constructor(
    options: Options<T>,
    private readonly labels: Labels<T>
  ) {
    super(options.fallback)
    this.members = options.members
    this.choices = options.members.map((value) => ({ value, ...labels[value] }))
  }

  get label(): string {
    return this.labels[this.value].label
  }
}

export class NumberField extends Field<number> {
  readonly minimum: number
  readonly maximum: number

  constructor(
    range: Range,
    readonly step: number,
    private readonly integer = false
  ) {
    super(range.default)
    this.minimum = range.minimum
    this.maximum = range.maximum
  }

  protected normalise(value: number): number {
    const held = Math.min(this.maximum, Math.max(this.minimum, value))
    return this.integer ? Math.round(held) : held
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
