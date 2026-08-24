import type { Meta } from '$lib/api'
import { must } from '$lib/invariant'
import { MORPHOLOGIES } from '$lib/labels'
import { IndexListField } from './fields.svelte'
import type { MosaicState } from './mosaic.svelte'
import { clausePoints } from '@uwdata/mosaic-core'
import { column } from '@uwdata/mosaic-sql'
import { onDestroy } from 'svelte'

export interface MorphologyClass {
  index: number
  label: string
  count: number
}

export class FilterState {
  readonly morphologies = new IndexListField()

  constructor(
    private readonly meta: Meta,
    private readonly mosaic: MosaicState
  ) {}

  get classes(): MorphologyClass[] {
    return [...this.meta.morphologies, this.meta.unlabelled].map((count, index) => ({
      index,
      label: must(MORPHOLOGIES[index], `name for morphology ${index}`),
      count
    }))
  }

  get active(): boolean {
    return this.morphologies.value.length > 0
  }

  #push(indices: number[]): void {
    this.mosaic.filter.update(
      clausePoints([column('category')], indices.length ? indices.map((index) => [index]) : null, {
        source: this
      })
    )
  }

  start(): void {
    $effect(() => this.#push(this.morphologies.value))
    onDestroy(() => this.#push([]))
  }
}
