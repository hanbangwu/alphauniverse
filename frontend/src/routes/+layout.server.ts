import { getMeta } from '$lib/api'
import type { Detail } from '$lib/api'
import type { LayoutServerLoad } from './$types'
import { error } from '@sveltejs/kit'

export const load: LayoutServerLoad = async ({ fetch }) => {
  const { data, error: failure } = await getMeta({ fetch })
  if (!data) {
    error(503, failure instanceof Error ? failure.message : (failure as Detail).detail)
  }
  return { meta: data }
}
