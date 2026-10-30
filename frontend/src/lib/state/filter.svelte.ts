import { Field } from './field.svelte'
import type { MosaicState } from './mosaic.svelte'
import { DEFAULTS, stored } from './params.svelte'
import type { ViewState } from './view.svelte'
import { clauseInterval } from '@uwdata/mosaic-core'
import { column } from '@uwdata/mosaic-sql'
import { onDestroy, untrack } from 'svelte'

export class FilterState {
  readonly range: Field<[number, number] | null>

  constructor(
    private readonly mosaic: MosaicState,
    private readonly view: ViewState
  ) {
    this.range = new Field(DEFAULTS.range, stored('range'))
  }

  #pushRange(range: [number, number] | null): void {
    this.mosaic.filter.update(clauseInterval(column('value'), range, { source: this.range }))
  }

  start(): void {
    $effect(() => this.#pushRange(this.range.value))
    const colouring = () => `${this.view.property.value} ${this.view.everywhere}`
    let shown = colouring()
    $effect(() => {
      if (colouring() === shown) return
      shown = colouring()
      untrack(() => this.range.reset())
    })
    onDestroy(() => this.#pushRange(null))
  }
}
