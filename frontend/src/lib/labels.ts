import type { Galaxy, TableRow } from '$lib/api'

export interface Label {
  label: string
}

export type Labels<T extends string> = Record<T, Label>

export type Choice<T extends string> = Label & { value: T }

export const SIMILARITY = { short: 'similarity' } as const

export const OBSERVATIONS: Record<string, string> = {
  hsc: 'HSC match',
  desi: 'DESI spectrum',
  sdss: 'SDSS spectrum',
  redshift: 'redshift'
}

export const SURVEYS: Labels<keyof Galaxy> = {
  ls: { label: 'Legacy Survey DR10' },
  hsc: { label: 'HSC PDR3' },
  desi: { label: 'DESI EDR SV3' },
  sdss: { label: 'SDSS' },
  gz10: { label: 'Galaxy Zoo 10' },
  provabgs: { label: 'PROVABGS' }
}

export const SECTIONS: Labels<TableRow['section']> = {
  redshift: { label: 'Redshift' },
  ...SURVEYS
}

export const POINT_SETS = {
  same_place: { label: 'Same place' },
  any_position: { label: 'Any position' },
  mean: { label: 'Mean' },
  full: { label: 'Full' }
} satisfies Labels<string>

export type PointSet = keyof typeof POINT_SETS

export const COLOURINGS = {
  morphology: { label: 'Morphology' },
  redshift: { label: 'Redshift' },
  mass: { label: 'Stellar mass' },
  sfr: { label: 'SFR' },
  ssfr: { label: 'sSFR' },
  g_r: { label: 'g − r' }
} satisfies Labels<string>

export type Colouring = keyof typeof COLOURINGS

export type Property = Exclude<Colouring, 'morphology'>

export const PREDICTABLE: readonly Property[] = ['redshift', 'mass', 'sfr', 'ssfr']

export const UNITS: Record<Property, string> = {
  redshift: 'z',
  mass: 'log M☉',
  sfr: 'log M☉ yr⁻¹',
  ssfr: 'log yr⁻¹',
  g_r: 'mag'
}

export const SOURCES = {
  measured: { label: 'Measured first' },
  predicted: { label: 'Predicted' }
} satisfies Labels<string>

export type Source = keyof typeof SOURCES

export const MORPHOLOGIES: readonly string[] = [
  'Disturbed Galaxies',
  'Merging Galaxies',
  'Round Smooth Galaxies',
  'In-between Round Smooth Galaxies',
  'Cigar Shaped Smooth Galaxies',
  'Barred Spiral Galaxies',
  'Unbarred Tight Spiral Galaxies',
  'Unbarred Loose Spiral Galaxies',
  'Edge-on Galaxies without Bulge',
  'Edge-on Galaxies with Bulge',
  'No label'
]

export const UNLABELLED: number = MORPHOLOGIES.length - 1
