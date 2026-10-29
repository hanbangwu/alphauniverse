import type { Galaxy, ShareName, TableRow } from '$lib/api'
import type { Projection } from '$lib/client'
import type { ModeName } from '$lib/data/similarity'

export interface Label {
  label: string
}

export type Labels<T extends string> = Record<T, Label>

export type Survey = Exclude<keyof Galaxy, 'object_id' | 'ra' | 'dec'>

export type Choice<T extends string> = Label & { value: T }

export const SIMILARITY = { short: 'similarity' } as const

export const OBSERVATIONS: Record<string, string> = {
  hsc: 'HSC match',
  desi: 'DESI spectrum',
  sdss: 'SDSS spectrum',
  redshift: 'redshift'
}

export const SURVEYS: Labels<Survey> = {
  ls: { label: 'Legacy Survey DR10' },
  hsc: { label: 'HSC PDR3' },
  desi: { label: 'DESI EDR SV3' },
  sdss: { label: 'SDSS' },
  gz10: { label: 'Galaxy Zoo 10' },
  provabgs: { label: 'PROVABGS' }
}

export const MODES: Labels<ModeName> = {
  ls_image: { label: 'Legacy Survey image' },
  hsc_image: { label: 'HSC image' },
  desi_spectrum: { label: 'DESI spectrum' },
  sdss_spectrum: { label: 'SDSS spectrum' },
  redshift: { label: 'Redshift' },
  ls_table: { label: 'Legacy Survey catalogue' },
  hsc_table: { label: 'HSC catalogue' }
}

export const SHARES: Labels<ShareName> = {
  ...MODES,
  combined: { label: 'Combined views' }
}

export const SECTIONS: Labels<TableRow['section']> = {
  ls: SURVEYS.ls,
  hsc: SURVEYS.hsc,
  desi: SURVEYS.desi,
  sdss: SURVEYS.sdss,
  redshift: { label: 'Redshift' },
  provabgs: SURVEYS.provabgs,
  gz10: SURVEYS.gz10
}

export const POINT_SETS = {
  any_position: { label: 'Any position' },
  same_place: { label: 'Same place' }
} satisfies Partial<Labels<Projection>>

export type PointSet = keyof typeof POINT_SETS

export const PROPERTIES = {
  redshift: { label: 'Redshift' },
  mass: { label: 'Stellar mass' },
  sfr: { label: 'SFR' },
  ssfr: { label: 'sSFR' },
  g_r: { label: 'g − r' },
  radius: { label: 'Half-light radius' },
  r_magnitude: { label: 'r magnitude' },
  axis_ratio: { label: 'Axis ratio' },
  halpha_ew: { label: 'Hα equivalent width' },
  dn4000: { label: 'Dn4000' }
} satisfies Labels<string>

export type Property = keyof typeof PROPERTIES

export const PREDICTABLE: readonly Property[] = (Object.keys(PROPERTIES) as Property[]).filter(
  (property) => property !== 'g_r'
)

export const UNITS: Record<Property, string> = {
  redshift: 'z',
  mass: 'log M☉',
  sfr: 'log M☉ yr⁻¹',
  ssfr: 'log yr⁻¹',
  g_r: 'mag',
  radius: 'arcsec',
  r_magnitude: 'mag',
  axis_ratio: 'b/a',
  halpha_ew: 'Å',
  dn4000: 'ratio'
}

export const SOURCES = {
  measured: { label: 'Measured first' },
  predicted: { label: 'Predicted' }
} satisfies Labels<string>

export type Source = keyof typeof SOURCES
