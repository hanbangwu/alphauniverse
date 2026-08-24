import { PUBLIC_API_URL } from '$env/static/public'
import { getMeta } from '$lib/api'
import type { LayoutServerLoad } from './$types'
import { error } from '@sveltejs/kit'

function describe(failure: unknown): string {
  if (failure instanceof Error) return failure.message
  if (typeof failure === 'object' && failure !== null && 'detail' in failure) {
    return String(failure.detail)
  }
  return failure === undefined ? 'no response' : JSON.stringify(failure)
}

export const load: LayoutServerLoad = async ({ fetch }) => {
  try {
    const { data, error: failure, response } = await getMeta({ fetch })
    if (data) return { meta: data }

    const status = response ? ` (HTTP ${response.status})` : ''
    error(503, `${PUBLIC_API_URL}/meta did not answer${status}: ${describe(failure)}`)
  } catch (failure) {
    if (typeof failure === 'object' && failure !== null && 'status' in failure) throw failure
    error(503, `${PUBLIC_API_URL}/meta is unreachable: ${describe(failure)}`)
  }
}
