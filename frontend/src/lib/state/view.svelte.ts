import type { Meta } from '$lib/api'
import { DETAIL_VIEWS, type DetailView, POINT_SETS, type PointSet } from '$lib/labels'
import { Field } from './field.svelte'
import { EnumField, IndexListField } from './fields.svelte'
import { optionsOver } from './schema'

export class ViewState {
  readonly pointSet: EnumField<PointSet>
  readonly galaxy = new Field<number | null>(null)
  readonly detail: EnumField<DetailView>
  readonly patches = new IndexListField()
  readonly spans = new IndexListField()
  readonly explorer = new Field(false)

  constructor(private readonly meta: Meta) {
    this.pointSet = new EnumField(optionsOver(POINT_SETS), POINT_SETS)
    this.detail = new EnumField(optionsOver(DETAIL_VIEWS), DETAIL_VIEWS)
  }

  get full(): boolean {
    return this.pointSet.value === 'full'
  }

  get table(): string {
    return this.full ? this.meta.full_points : this.meta.mean_points
  }

  select(galaxy: number | null): void {
    if (galaxy !== this.galaxy.value) {
      this.patches.reset()
      this.spans.reset()
    }
    this.galaxy.value = galaxy
  }
}
