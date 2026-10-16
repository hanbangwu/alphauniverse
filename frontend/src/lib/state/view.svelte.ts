import { DETAIL_VIEWS, type DetailView, POINT_SETS, type PointSet } from '$lib/labels'
import { Field } from './field.svelte'
import { EnumField, IndexListField } from './fields.svelte'

export class ViewState {
  readonly pointSet: EnumField<PointSet>
  readonly galaxy = new Field<number | null>(null)
  readonly detail: EnumField<DetailView>
  readonly imageTokens = new IndexListField()
  readonly spectrumTokens = new IndexListField()
  readonly tableValues = new IndexListField()
  readonly imageAnywhere = new Field(false)
  readonly spectrumAnywhere = new Field(false)
  readonly explorer = new Field(false)

  constructor() {
    this.pointSet = new EnumField(POINT_SETS)
    this.detail = new EnumField(DETAIL_VIEWS)
  }

  get full(): boolean {
    return this.pointSet.value === 'full'
  }

  get table(): string {
    return this.full ? 'full_points' : 'mean_points'
  }

  select(galaxy: number | null): void {
    if (galaxy !== this.galaxy.value) {
      this.imageTokens.reset()
      this.spectrumTokens.reset()
      this.tableValues.reset()
    }
    this.galaxy.value = galaxy
  }
}
