import { byMode } from '$lib/data/modes'
import type { SimilarityQuery } from '$lib/data/similarity'
import {
  COLOURINGS,
  type Colouring,
  POINT_SETS,
  PREDICTABLE,
  type PointSet,
  type Property,
  SOURCES,
  type Source
} from '$lib/labels'
import { Field } from './field.svelte'
import { EnumField, IndexListField } from './fields.svelte'
import { PersistedState } from 'runed'

export class ViewState {
  readonly pointSet: EnumField<PointSet>
  readonly colouring: EnumField<Colouring> = new EnumField(COLOURINGS)
  readonly source: EnumField<Source> = new EnumField(SOURCES)
  readonly galaxy = new Field<number | null>(null)
  readonly tokens = byMode(() => new IndexListField())
  readonly tableValues = new IndexListField()
  readonly imageAnywhere = new Field(false)
  readonly spectrumAnywhere = new Field(false)
  readonly searched = new Field<SimilarityQuery | null>(null)
  readonly explorer = new Field(false)
  readonly predictions = new Field(false)
  readonly open = new PersistedState('open-groups', {
    images: true,
    spectra: true,
    catalogue: true
  })

  constructor() {
    this.pointSet = new EnumField(POINT_SETS)
  }

  get full(): boolean {
    return this.pointSet.value === 'full'
  }

  get property(): Property | null {
    const colouring = this.colouring.value
    return colouring === 'morphology' ? null : colouring
  }

  get predictable(): boolean {
    return this.property !== null && PREDICTABLE.includes(this.property)
  }

  get everywhere(): boolean {
    return this.predictable && this.source.value === 'predicted'
  }

  select(galaxy: number | null): void {
    if (galaxy !== this.galaxy.value) {
      for (const field of Object.values(this.tokens)) field.reset()
      this.tableValues.reset()
      this.searched.reset()
    }
    this.galaxy.value = galaxy
  }
}
