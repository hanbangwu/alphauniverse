export interface Source<T> {
  get(): T
  set(value: T): void
}

export class Field<T> {
  #value: T = $state()!

  constructor(
    readonly fallback: T,
    private readonly source?: Source<T>
  ) {
    this.#value = fallback
  }

  get value(): T {
    return this.source ? this.source.get() : this.#value
  }

  set value(next: T) {
    const value = this.normalise(next)
    if (this.source) this.source.set(value)
    else this.#value = value
  }

  reset(): void {
    this.value = this.fallback
  }

  protected normalise(value: T): T {
    return value
  }
}
