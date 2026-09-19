import { dev } from '$app/environment'
import type { CreateClientConfig } from './api/client.gen'

export const createClientConfig: CreateClientConfig = (config) => ({
  ...config,
  baseUrl: dev ? 'http://127.0.0.1:8000' : 'https://surp-2026--alphauniverse-fastapi-app.modal.run'
})
