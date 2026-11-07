import { browser } from '$app/environment'
import type { CreateClientConfig } from './api/client.gen'

export const createClientConfig: CreateClientConfig = (config) => ({
  ...config,
  baseUrl: browser ? `${location.origin}/api` : '/api'
})
