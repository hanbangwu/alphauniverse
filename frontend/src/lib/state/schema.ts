import { zGetSimilarityQuery } from '$lib/api/zod.gen'
import { must } from '$lib/invariant'
import type { Labels } from '$lib/labels'
import * as z from 'zod'

export type Param = keyof typeof zGetSimilarityQuery.shape

export interface Range {
  minimum: number
  maximum: number
  default: number
}

export interface Options<T extends string> {
  members: readonly T[]
  fallback: T
}

const DEFAULTS: Record<string, unknown> = zGetSimilarityQuery.parse({ galaxy: 0, p: [0] })

export function rangeOf(name: Param): Range {
  let schema: unknown = zGetSimilarityQuery.shape[name]
  while (schema instanceof z.ZodDefault || schema instanceof z.ZodOptional) {
    schema = schema.unwrap()
  }
  if (!(schema instanceof z.ZodNumber)) throw new Error(`${name} is not a number`)
  return {
    minimum: must(schema.minValue, `minimum for ${name}`),
    maximum: must(schema.maxValue, `maximum for ${name}`),
    default: must(DEFAULTS[name], `default for ${name}`) as number
  }
}

export function optionsOver<T extends string>(labels: Labels<T>): Options<T> {
  const members = Object.keys(labels) as T[]
  return { members, fallback: must(members[0], 'a first member to fall back to') }
}
