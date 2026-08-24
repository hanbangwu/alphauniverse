import { defaultPlugins, defineConfig } from '@hey-api/openapi-ts'

export default defineConfig({
  input: './openapi.json',
  output: {
    path: './src/lib/api',
    postProcess: ['prettier']
  },
  plugins: [
    ...defaultPlugins,
    { name: '@hey-api/typescript', enums: 'javascript' },
    {
      name: '@hey-api/client-fetch',
      baseUrl: false,
      runtimeConfigPath: './src/lib/hey-api.ts'
    },
    { name: 'zod', definitions: false, responses: false }
  ]
})
