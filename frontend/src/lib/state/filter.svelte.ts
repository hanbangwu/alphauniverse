import { Field } from './field.svelte'
import type { MosaicState } from './mosaic.svelte'
import type { ViewState } from './view.svelte'
import { clauseInterval } from '@uwdata/mosaic-core'
import { column } from '@uwdata/mosaic-sql'
import { onDestroy } from 'svelte'

export class FilterState {
  readonly range = new Field<[number, number] | null>(null)

  constructor(
    private readonly mosaic: MosaicState,
    private readonly view: ViewState
  ) {}

  #pushRange(range: [number, number] | null): void {
    this.mosaic.filter.update(clauseInterval(column('value'), range, { source: this.range }))
  }

  start(): void {
    $effect(() => this.#pushRange(this.range.value))
    $effect(() => {
      void [this.view.property.value, this.view.everywhere]
      this.range.reset()
    })
    onDestroy(() => this.#pushRange(null))
  }
}
