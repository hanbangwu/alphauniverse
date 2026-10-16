import { dev } from '$app/environment'
import { env } from '$env/dynamic/public'
import type { CreateClientConfig } from './api/client.gen'

export const createClientConfig: CreateClientConfig = (config) => ({
  ...config,
  baseUrl: dev ? 'http://127.0.0.1:8000' : env.PUBLIC_API_URL
})
