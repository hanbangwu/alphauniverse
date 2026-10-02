import { artifactUrl } from '$lib/client'
import { UNLABELLED } from '$lib/labels'
import { Coordinator, Selection, wasmConnector } from '@uwdata/mosaic-core'
import { loadParquet } from '@uwdata/mosaic-sql'

export class MosaicState {
  readonly coordinator = new Coordinator()
  readonly filter = Selection.intersect()

  #connected = false

  connect(): Coordinator {
    if (!this.#connected) {
      this.coordinator.databaseConnector(wasmConnector())
      this.#connected = true
    }
    return this.coordinator
  }

  async load(role: string): Promise<string> {
    const database = this.connect()
    await database.exec(
      loadParquet(role, artifactUrl(role), {
        replace: true,
        select: ['* EXCLUDE (category)', `coalesce(category, ${UNLABELLED})::UTINYINT AS category`]
      })
    )
    database.clear({ clients: false, cache: true })
    return role
  }

  destroy(): void {
    this.coordinator.clear()
  }
}
