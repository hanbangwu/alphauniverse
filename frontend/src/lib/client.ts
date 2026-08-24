import type { GetArtifactData } from './api'
import { client } from './api/client.gen'

export { client }

export function artifactUrl(role: string): string {
  return client.buildUrl<GetArtifactData>({
    url: '/artifacts/{role}',
    path: { role }
  })
}
