import type { DownloadArtifactData, GetProjectionsData } from './api'
import { client } from './api/client.gen'

export { client }

export type Projection = GetProjectionsData['path']['projection']

export function projectionUrl(projection: Projection): string {
  return client.buildUrl<GetProjectionsData>({
    url: '/projections/{projection}',
    path: { projection }
  })
}

export function downloadUrl(role: DownloadArtifactData['path']['role']): string {
  return client.buildUrl<DownloadArtifactData>({
    url: '/downloads/{role}',
    path: { role }
  })
}
