import type { TokenMode } from '$lib/client'
import { PATCH_MODES, byMode } from '$lib/data/modes'
import type { SearchQuery } from '$lib/data/similarity'
import { MODES, POINT_SETS, PROPERTIES } from '$lib/labels'
import type { Source } from './field.svelte'
import { rangeOf } from './schema'
import { Context } from 'runed'
import { type ReturnUseSearchParams, useSearchParams } from 'runed/kit'
import { z } from 'zod'

const choice = <T extends string>(labels: Record<T, unknown>) => {
  const members = Object.keys(labels) as [T, ...T[]]
  return z.enum(members).catch(members[0])
}

const flag = z.boolean().catch(false)
const indices = z.array(z.int().nonnegative()).catch([])
const threshold = z.coerce.number().nullable().catch(null)
const matches = rangeOf('matches')

export const schema = z.object({
  projection: choice(POINT_SETS),
  property: choice(PROPERTIES),
  range: z
    .tuple([z.number(), z.number()])
    .refine(([low, high]) => low <= high)
    .nullable()
    .catch(null),
  region: z
    .union([
      z.object({ xMin: z.number(), xMax: z.number(), yMin: z.number(), yMax: z.number() }),
      z.array(z.object({ x: z.number(), y: z.number() })).min(3)
    ])
    .nullable()
    .catch(null),
  galaxy: z.string().min(1).nullable().catch(null),
  explorer: flag,
  data_source: z.enum(['dataset', 'prediction']).catch('dataset'),
  search_by: z.enum(['patch', 'whole']).catch('patch'),
  mask: flag,
  invert: flag,
  image_false_color: flag,
  spectrum_false_color: flag,
  image_threshold: threshold,
  image_match_threshold: threshold,
  spectrum_threshold: threshold,
  spectrum_match_threshold: threshold,
  table_threshold: threshold,
  tab: z.enum(['sky', 'umap', 'text']).catch('sky'),
  text: z.string().nullable().catch(null),
  ...byMode(() => indices),
  table_values: indices,
  anywhere: z.array(z.enum(PATCH_MODES as [TokenMode, ...TokenMode[]])).catch([]),
  modes: z.array(z.enum(Object.keys(MODES) as [keyof typeof MODES])).catch([]),
  matches: z.int().min(matches.minimum).max(matches.maximum).catch(matches.default)
})

type Params = ReturnUseSearchParams<typeof schema>

export type Values = z.output<typeof schema>

export const DEFAULTS = schema.parse({})

const context = new Context<Params>('params')

export const setParams = (): Params =>
  context.set(useSearchParams(schema, { pushHistory: false, noScroll: true }))

export function stored<K extends keyof Values>(key: K): Source<Values[K]> {
  const params: Values = context.get()
  return {
    get: () => params[key],
    set: (next) => {
      params[key] = next
    }
  }
}

export function searched(): Source<SearchQuery | null> {
  const params = context.get()
  const query = $derived.by((): SearchQuery | null => {
    const galaxy = params.galaxy
    if (galaxy === null) return null
    const common = { galaxy, anywhere: params.anywhere, matches: params.matches }
    if (params.modes.length) return { ...common, modes: params.modes }
    const lists = byMode((mode) => params[mode])
    if (!params.table_values.length && PATCH_MODES.every((mode) => !lists[mode].length)) return null
    return { ...common, ...lists, table_values: params.table_values }
  })
  return {
    get: () => query,
    set: (next) =>
      params.update({
        ...byMode(() => []),
        table_values: [],
        anywhere: [],
        modes: [],
        matches: DEFAULTS.matches,
        ...next
      })
  }
}
