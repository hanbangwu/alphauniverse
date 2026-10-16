import type { GetSearchData } from '$lib/api'
import { must } from '$lib/invariant'
import { paths } from '../../../openapi.json'

export type Param = keyof NonNullable<GetSearchData['query']>

export interface Range {
  minimum: number
  maximum: number
  default: number
}

export function rangeOf(name: Param): Range {
  const { schema } = must(
    paths['/search'].get.parameters.find((parameter) => parameter.name === name),
    `parameter ${name}`
  )
  if (typeof schema.default !== 'number') throw new Error(`${name} has no number default`)
  return {
    minimum: must(schema.minimum, `minimum for ${name}`),
    maximum: must(schema.maximum, `maximum for ${name}`),
    default: schema.default
  }
}
