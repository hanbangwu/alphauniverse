import type { GetCosinesData, GetImageData, GetProjectionsData, GetSpectrumData } from './api'
import type { HttpValidationError } from './api'
import { login } from './api'
import { client } from './api/client.gen'

export { client }

client.interceptors.error.use((error, response) => {
  if (error instanceof Error) return error
  const { detail } = (error ?? {}) as { detail?: string | HttpValidationError['detail'] }
  if (typeof detail === 'string') return new Error(detail)
  if (detail) return new Error(detail.map(({ msg }) => msg).join('; '))
  return new Error(response?.statusText || 'The request failed.')
})

export type Projection = GetProjectionsData['path']['projection']

export type CosineMode = GetCosinesData['path']['mode']

export type ImageSurvey = GetImageData['path']['survey']

export type SpectrumSurvey = GetSpectrumData['path']['survey']

export async function signIn(password: string): Promise<Response | undefined> {
  const { response } = await login({ headers: { Authorization: `Bearer ${password}` } })
  return response
}
