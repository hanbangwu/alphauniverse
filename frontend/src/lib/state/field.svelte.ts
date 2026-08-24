export class Field<T> {
  #value: T = $state()!

  constructor(readonly fallback: T) {
    this.#value = fallback
  }

  get value(): T {
    return this.#value
  }

  set value(next: T) {
    this.#value = this.normalise(next)
  }

  reset(): void {
    this.#value = this.fallback
  }

  protected normalise(value: T): T {
    return value
  }
}
