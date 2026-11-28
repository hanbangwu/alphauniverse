import type { GalaxyId } from '$lib/client'
import { IMAGE_MODES, SPECTRUM_MODES, PATCH_MODES, byMode } from '$lib/data/modes'
import { type ModeName, type SearchQuery, isModeQuery } from '$lib/data/similarity'
import { POINT_SETS, PROPERTIES, type PointSet, type Property, UNITS } from '$lib/labels'
import { Field } from './field.svelte'
import { EnumField, IndexListField, ListField } from './fields.svelte'
import type { MosaicState } from './mosaic.svelte'
import { DEFAULTS, type Values, searched, stored } from './params.svelte'
import { PersistedState } from 'runed'

export class ViewState {
  readonly tab: Field<Values['tab']>
  readonly pointSet: EnumField<PointSet>
  readonly property: EnumField<Property>
  readonly galaxy: Field<GalaxyId | null>
  readonly patches = byMode(() => new IndexListField())
  readonly tableValues = new IndexListField()
  readonly imageAnywhere = new Field(false)
  readonly spectrumAnywhere = new Field(false)
  readonly searchBy: Field<Values['search_by']>
  readonly modes = new ListField<ModeName>()
  readonly searched: Field<SearchQuery | null>
  readonly explorer: Field<boolean>
  readonly dataSource: Field<Values['data_source']>
  readonly open = new PersistedState('open-sections', {
    images: true,
    spectra: false,
    catalog: false
  })

  constructor(private readonly mosaic: MosaicState) {
    this.tab = new Field(DEFAULTS.tab, stored('tab'))
    this.pointSet = new EnumField(POINT_SETS, stored('projection'))
    this.property = new EnumField(
      Object.fromEntries(
        Object.entries(PROPERTIES).map(([key, label]) => [
          key,
          `${label} (${UNITS[key as Property]})`
        ])
      ) as Record<Property, string>,
      stored('property')
    )
    this.galaxy = new Field(DEFAULTS.galaxy, stored('galaxy'))
    this.searchBy = new Field(DEFAULTS.search_by, stored('search_by'))
    this.searched = new Field<SearchQuery | null>(null, searched())
    this.explorer = new Field(DEFAULTS.explorer, stored('explorer'))
    this.dataSource = new Field(DEFAULTS.data_source, stored('data_source'))
  }

  get predictable(): boolean {
    return this.mosaic.predictable.has(this.property.value)
  }

  get everywhere(): boolean {
    return this.predictable && this.dataSource.value === 'prediction'
  }

  edit(searched: SearchQuery): void {
    const anywhere = searched.anywhere ?? []
    this.imageAnywhere.value = IMAGE_MODES.some((mode) => anywhere.includes(mode))
    this.spectrumAnywhere.value = SPECTRUM_MODES.some((mode) => anywhere.includes(mode))
    if (isModeQuery(searched)) {
      this.modes.value = searched.modes
      return
    }
    for (const mode of PATCH_MODES) this.patches[mode].value = searched[mode] ?? []
    this.tableValues.value = searched.table_values ?? []
  }

  select(galaxy: GalaxyId | null): void {
    if (galaxy !== this.galaxy.value) {
      for (const field of Object.values(this.patches)) field.reset()
      this.tableValues.reset()
      this.modes.reset()
      this.searched.reset()
    }
    this.galaxy.value = galaxy
  }
}
