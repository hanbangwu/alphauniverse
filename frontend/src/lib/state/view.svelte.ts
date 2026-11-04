import { IMAGE_MODES, SPECTRUM_MODES, TOKEN_MODES, byMode } from '$lib/data/modes'
import { type ModeName, type SearchQuery, isModeQuery } from '$lib/data/similarity'
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
import { DEFAULTS, searched, stored } from './params.svelte'
import { PersistedState } from 'runed'

export class ViewState {
  readonly pointSet: EnumField<PointSet>
  readonly property: EnumField<Property>
  readonly source: EnumField<Source>
  readonly galaxy: Field<number | null>
  readonly tokens = byMode(() => new IndexListField())
  readonly tableValues = new IndexListField()
  readonly imageAnywhere = new Field(false)
  readonly spectrumAnywhere = new Field(false)
  readonly wholeModes: Field<boolean>
  readonly modes = new ListField<ModeName>()
  readonly searched: Field<SearchQuery | null>
  readonly explorer: Field<boolean>
  readonly predictions: Field<boolean>
  readonly open = new PersistedState('open-groups', {
    images: true,
    spectra: true,
    catalogue: true
  })

  constructor() {
    this.pointSet = new EnumField(POINT_SETS, stored('projection'))
    this.property = new EnumField(PROPERTIES, stored('property'))
    this.source = new EnumField(SOURCES, stored('source'))
    this.galaxy = new Field(DEFAULTS.galaxy, stored('galaxy'))
    this.wholeModes = new Field(DEFAULTS.whole, stored('whole'))
    this.searched = new Field<SearchQuery | null>(null, searched())
    this.explorer = new Field(DEFAULTS.explorer, stored('explorer'))
    this.predictions = new Field(DEFAULTS.predictions, stored('predictions'))
  }

  get predictable(): boolean {
    return PREDICTABLE.includes(this.property.value)
  }

  get everywhere(): boolean {
    return this.predictable && this.source.value === 'predicted'
  }

  edit(searched: SearchQuery): void {
    const anywhere = searched.anywhere ?? []
    this.imageAnywhere.value = IMAGE_MODES.some((mode) => anywhere.includes(mode))
    this.spectrumAnywhere.value = SPECTRUM_MODES.some((mode) => anywhere.includes(mode))
    if (isModeQuery(searched)) {
      this.modes.value = searched.modes
      return
    }
    for (const mode of TOKEN_MODES) this.tokens[mode].value = searched[mode] ?? []
    this.tableValues.value = searched.table_values ?? []
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
