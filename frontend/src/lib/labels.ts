export interface Label {
  label: string
}

export type Labels<T extends string> = Record<T, Label>

export type Choice<T extends string> = Label & { value: T }

export const SIMILARITY = { label: 'Cosine similarity', short: 'cos sim' } as const

export const SURVEYS: Labels<string> = {
  ls: { label: 'Legacy Survey DR10' },
  hsc: { label: 'HSC PDR3' },
  desi: { label: 'DESI EDR SV3' },
  sdss: { label: 'SDSS' },
  gz10: { label: 'Galaxy Zoo 10' },
  provabgs: { label: 'PROVABGS' }
}

export const POINT_SETS = {
  mean: { label: 'Mean' },
  full: { label: 'Full' }
} satisfies Labels<string>

export const DETAIL_VIEWS = {
  image: { label: 'Image' },
  tokens: { label: 'Tokens' },
  spectrum: { label: 'Spectrum' }
} satisfies Labels<string>

export type PointSet = keyof typeof POINT_SETS
export type DetailView = keyof typeof DETAIL_VIEWS

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

export function checkMorphologies(classes: number): void {
  if (classes !== UNLABELLED) {
    throw new Error(`the API counts ${classes} morphologies, this build names ${UNLABELLED}`)
  }
}
