import { byMode } from '$lib/data/modes'
import type { SimilarityQuery } from '$lib/data/similarity'
import { POINT_SETS, type PointSet } from '$lib/labels'
import { Field } from './field.svelte'
import { EnumField, IndexListField } from './fields.svelte'
import { PersistedState } from 'runed'

export class ViewState {
  readonly pointSet: EnumField<PointSet>
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

  get table(): string {
    return this.full ? 'full_points' : 'mean_points'
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
