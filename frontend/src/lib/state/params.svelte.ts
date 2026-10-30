import { goto } from '$app/navigation'
import { resolve } from '$app/paths'
import { page } from '$app/state'
import type { CosineMode } from '$lib/client'
import { TOKEN_MODES, byMode } from '$lib/data/modes'
import type { SearchQuery } from '$lib/data/similarity'
import { MODES, POINT_SETS, PROPERTIES, SOURCES } from '$lib/labels'
import type { Source } from './field.svelte'
import { rangeOf } from './schema'
import { untrack } from 'svelte'
import { SvelteURLSearchParams } from 'svelte/reactivity'
import { z } from 'zod'

const index = z.codec(z.string().regex(z.regexes.integer), z.int().nonnegative(), {
  decode: Number,
  encode: String
})

const decimal = z.codec(z.string().regex(z.regexes.number), z.number(), {
  decode: Number,
  encode: String
})

const list = <T extends z.ZodType<unknown[], string[]>>(items: T) =>
  z.codec(z.string(), items, {
    decode: (text) => text.split(',') as z.input<T>,
    encode: (values) => values.join(',')
  })

const json = <T extends z.core.$ZodType>(schema: T) =>
  z.codec(z.string(), schema, {
    decode: (text, ctx) => {
      try {
        return JSON.parse(text)
      } catch (error) {
        ctx.issues.push({
          code: 'invalid_format',
          format: 'json',
          input: text,
          message: (error as Error).message
        })
        return z.NEVER
      }
    },
    encode: (value) => JSON.stringify(value)
  })

const choice = <T extends string>(labels: Record<T, unknown>) => {
  const members = Object.keys(labels) as [T, ...T[]]
  return z.enum(members).catch(members[0])
}

const flag = z.stringbool().catch(false)
const indices = list(z.array(index)).catch([])
const threshold = decimal.nullable().catch(null)
const matches = rangeOf('matches')

export const schema = z.object({
  projection: choice(POINT_SETS),
  property: choice(PROPERTIES),
  source: choice(SOURCES),
  range: json(z.tuple([z.number(), z.number()]).refine(([low, high]) => low <= high))
    .nullable()
    .catch(null),
  region: json(
    z.union([
      z.object({ xMin: z.number(), xMax: z.number(), yMin: z.number(), yMax: z.number() }),
      z.array(z.object({ x: z.number(), y: z.number() })).min(3)
    ])
  )
    .nullable()
    .catch(null),
  galaxy: index.nullable().catch(null),
  explorer: flag,
  predictions: flag,
  whole: flag,
  layer: z.enum(['galaxy', 'population']).catch('galaxy'),
  mask: flag,
  invert: flag,
  image_threshold: threshold,
  image_match_threshold: threshold,
  spectrum_threshold: threshold,
  spectrum_match_threshold: threshold,
  table_threshold: threshold,
  tab: z.enum(['umap', 'text']).catch('umap'),
  text: z.string().nullable().catch(null),
  ...byMode(() => indices),
  table_values: indices,
  anywhere: list(z.array(z.enum(TOKEN_MODES as [CosineMode, ...CosineMode[]]))).catch([]),
  modes: list(z.array(z.enum(Object.keys(MODES) as [keyof typeof MODES]))).catch([]),
  matches: index.pipe(z.int().min(matches.minimum).max(matches.maximum)).catch(matches.default)
})

type Shape = typeof schema.shape

export type Values = z.output<typeof schema>

export const DEFAULTS = schema.parse({})

let pending = $state<SvelteURLSearchParams | null>(null)

function encode(key: keyof Shape, value: unknown): unknown {
  const field: z.ZodType = schema.shape[key]
  return z.encode(field, value)
}

export function write(changes: Partial<Values>): void {
  const params = new SvelteURLSearchParams(untrack(() => pending ?? page.url.searchParams))
  for (const key of Object.keys(changes) as (keyof Shape)[]) {
    const text = encode(key, changes[key])
    if (typeof text === 'string' && text !== encode(key, DEFAULTS[key])) params.set(key, text)
    else params.delete(key)
  }
  pending = params
  void goto(resolve(`/?${params}`), {
    replaceState: true,
    keepFocus: true,
    noScroll: true
  }).finally(() => {
    if (pending === params) pending = null
  })
}

export function stored<K extends keyof Shape>(key: K): Source<Values[K]> {
  const text = $derived((pending ?? page.url.searchParams).get(key))
  const value = $derived(z.parse(schema.shape[key], text ?? undefined) as Values[K])
  return {
    get: () => value,
    set: (next) => write({ [key]: next })
  }
}

export function searched(): Source<SearchQuery | null> {
  const galaxy = stored('galaxy')
  const tokens = byMode((mode) => stored(mode))
  const tableValues = stored('table_values')
  const anywhere = stored('anywhere')
  const modes = stored('modes')
  const count = stored('matches')
  const query = $derived.by((): SearchQuery | null => {
    const at = galaxy.get()
    if (at === null) return null
    const common = { galaxy: at, anywhere: anywhere.get(), matches: count.get() }
    if (modes.get().length) return { ...common, modes: modes.get() }
    const lists = byMode((mode) => tokens[mode].get())
    const table_values = tableValues.get()
    if (!table_values.length && TOKEN_MODES.every((mode) => !lists[mode].length)) return null
    return { ...common, ...lists, table_values }
  })
  return {
    get: () => query,
    set: (next) =>
      write({
        ...byMode(() => []),
        table_values: [],
        anywhere: [],
        modes: [],
        matches: DEFAULTS.matches,
        ...next
      })
  }
}
