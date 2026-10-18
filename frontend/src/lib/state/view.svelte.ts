import { byMode } from '$lib/data/modes'
import type { ModeName, SearchQuery } from '$lib/data/similarity'
import {
  POINT_SETS,
  PREDICTABLE,
  PROPERTIES,
  type PointSet,
  type Property,
  SOURCES,
  type Source
} from '$lib/labels'
import { Field } from './field.svelte'
import { EnumField, IndexListField, ListField } from './fields.svelte'
import { PersistedState } from 'runed'

export class ViewState {
  readonly pointSet: EnumField<PointSet>
  readonly property: EnumField<Property> = new EnumField(PROPERTIES)
  readonly source: EnumField<Source> = new EnumField(SOURCES)
  readonly galaxy = new Field<number | null>(null)
  readonly tokens = byMode(() => new IndexListField())
  readonly tableValues = new IndexListField()
  readonly imageAnywhere = new Field(false)
  readonly spectrumAnywhere = new Field(false)
  readonly wholeModes = new Field(false)
  readonly modes = new ListField<ModeName>()
  readonly searched = new Field<SearchQuery | null>(null)
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

  get predictable(): boolean {
    return PREDICTABLE.includes(this.property.value)
  }

  get everywhere(): boolean {
    return this.predictable && this.source.value === 'predicted'
  }

  select(galaxy: number | null): void {
    if (galaxy !== this.galaxy.value) {
      for (const field of Object.values(this.tokens)) field.reset()
      this.tableValues.reset()
      this.modes.reset()
      this.searched.reset()
    }
    this.galaxy.value = galaxy
  }
}
