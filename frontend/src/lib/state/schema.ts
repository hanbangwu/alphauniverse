import { zGetSearchQuery } from '$lib/api/zod.gen'
import { must } from '$lib/invariant'
import * as z from 'zod'

export type Param = keyof typeof zGetSearchQuery.shape

export interface Range {
  minimum: number
  maximum: number
  default: number
}

const DEFAULTS: Record<string, unknown> = zGetSearchQuery.parse({ galaxy: 0, p: [0] })

export function rangeOf(name: Param): Range {
  let schema: unknown = zGetSearchQuery.shape[name]
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
