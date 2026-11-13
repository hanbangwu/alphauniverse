import type { Galaxy, TableRow } from '$lib/api'
import type { Projection } from '$lib/client'
import type { ModeName } from '$lib/data/similarity'

export type Survey = Exclude<keyof Galaxy, 'object_id' | 'ra' | 'dec'>

export const SURVEYS: Record<Survey, string> = {
  ls: 'Legacy Survey DR10',
  hsc: 'HSC PDR3',
  desi: 'DESI EDR SV3',
  sdss: 'SDSS',
  gz10: 'Galaxy Zoo 10',
  provabgs: 'PROVABGS'
}

export const MODES: Record<ModeName, string> = {
  ls_image: 'Legacy Survey image',
  hsc_image: 'HSC image',
  desi_spectrum: 'DESI spectrum',
  sdss_spectrum: 'SDSS spectrum',
  redshift: 'Redshift',
  ls_table: 'Legacy Survey catalogue',
  hsc_table: 'HSC catalogue'
}

export const SECTIONS: Record<TableRow['section'], string> = { ...SURVEYS, redshift: 'Redshift' }

export const POINT_SETS = {
  any_position: 'Any position',
  same_place: 'Same place'
} satisfies Partial<Record<Projection, string>>

export type PointSet = keyof typeof POINT_SETS

export const PROPERTIES = {
  redshift: 'Redshift',
  mass: 'Stellar mass',
  sfr: 'SFR',
  ssfr: 'sSFR',
  g_r: 'g − r',
  radius: 'Half-light radius',
  r_magnitude: 'r magnitude',
  axis_ratio: 'Axis ratio',
  halpha_ew: 'Hα equivalent width',
  dn4000: 'Dn4000'
}

export type Property = keyof typeof PROPERTIES

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
  measured: 'Measured first',
  predicted: 'Predicted'
}

export type Source = keyof typeof SOURCES
