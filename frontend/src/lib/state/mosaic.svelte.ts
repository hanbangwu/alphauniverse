import { artifactUrl } from '$lib/client'
import { UNLABELLED } from '$lib/labels'
import { Coordinator, Selection, wasmConnector } from '@uwdata/mosaic-core'
import { loadParquet } from '@uwdata/mosaic-sql'
import { SvelteMap } from 'svelte/reactivity'

export class MosaicState {
  readonly coordinator = new Coordinator()
  readonly filter = Selection.intersect()

  #connected = false
  readonly #loading = new SvelteMap<string, Promise<string>>()

  connect(): Coordinator {
    if (!this.#connected) {
      this.coordinator.databaseConnector(wasmConnector())
      this.#connected = true
    }
    return this.coordinator
  }

  load(role: string): Promise<string> {
    let pending = this.#loading.get(role)
    if (!pending) {
      pending = this.#read(role).catch((error: unknown) => {
        this.#loading.delete(role)
        throw error
      })
      this.#loading.set(role, pending)
    }
    return pending
  }

  async #read(role: string): Promise<string> {
    const db = this.connect()
    await db.exec(
      loadParquet(role, artifactUrl(role), {
        replace: true,
        select: ['* EXCLUDE (category)', `coalesce(category, ${UNLABELLED})::UTINYINT AS category`]
      })
    )
    db.clear({ clients: false, cache: true })
    return role
  }

  destroy(): void {
    this.#loading.clear()
    this.coordinator.clear()
  }
}
