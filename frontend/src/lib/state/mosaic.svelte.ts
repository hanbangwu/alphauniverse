import { type Projection, downloadUrl, projectionUrl } from '$lib/client'
import { BINS } from '$lib/color'
import { type Property, type Source, UNLABELLED } from '$lib/labels'
import { Coordinator, Selection, wasmConnector } from '@uwdata/mosaic-core'
import { loadParquet } from '@uwdata/mosaic-sql'

export interface Coloured {
  table: string
  low: number
  high: number
  measured: number
  predicted: number
  missing: number
}

export class MosaicState {
  readonly coordinator = new Coordinator()
  readonly filter = Selection.intersect()

  #connected = false
  #properties: Promise<unknown> | null = null

  connect(): Coordinator {
    if (!this.#connected) {
      this.coordinator.databaseConnector(wasmConnector())
      this.#connected = true
    }
    return this.coordinator
  }

  async load(projection: Projection): Promise<string> {
    const table = `${projection}_points`
    const database = this.connect()
    await database.exec(
      loadParquet(table, projectionUrl(projection), {
        replace: true,
        select: ['* EXCLUDE (category)', `coalesce(category, ${UNLABELLED})::UTINYINT AS category`]
      })
    )
    database.clear({ clients: false, cache: true })
    return table
  }

  async colour(points: string, property: Property, source: Source): Promise<Coloured> {
    const database = this.connect()
    this.#properties ??= database.exec(
      loadParquet('properties', downloadUrl('properties'), { replace: true })
    )
    await this.#properties
    const photometric = property === 'g_r'
    const everywhere = source === 'predicted' && !photometric
    const measured = photometric ? 'g_r' : `${property}_measured`
    const shown = photometric
      ? measured
      : everywhere
        ? `${property}_predicted`
        : `coalesce(${measured}, ${property}_predicted)`
    const predicted = photometric ? 'false' : everywhere ? 'true' : `${measured} IS NULL`
    const width = photometric ? 'NULL' : `${property}_width`
    const values = `${property}_${everywhere ? 'predicted' : 'measured'}`
    await database.exec(`
      CREATE OR REPLACE TABLE ${values} AS
      SELECT galaxy, ${shown} AS value, ${predicted} AS predicted,
        CASE WHEN ${predicted} THEN ${width} END AS width
      FROM properties`)
    const summary = (
      await database.query(`
        SELECT quantile_cont(value, 0.01) AS low, quantile_cont(value, 0.99) AS high,
          count(value) FILTER (WHERE NOT predicted) AS measured,
          count(value) FILTER (WHERE predicted) AS predicted,
          count(*) - count(value) AS missing
        FROM ${values}`)
    ).get(0)
    const low = Number(summary.low)
    const high = Number(summary.high)
    const span = high > low ? high - low : 1
    const faded = everywhere ? '0' : `CASE WHEN c.predicted THEN ${BINS} ELSE 0 END`
    const table = `${points}_${values}`
    await database.exec(`
      CREATE OR REPLACE VIEW ${table} AS
      SELECT p.*, c.value, c.width, c.predicted,
        coalesce(
          least(${BINS - 1}, greatest(0, floor((c.value - ${low}) / ${span} * ${BINS})))::UTINYINT + ${faded},
          ${2 * BINS}
        )::UTINYINT AS colour
      FROM ${points} p LEFT JOIN ${values} c USING (galaxy)`)
    database.clear({ clients: false, cache: true })
    return {
      table,
      low,
      high,
      measured: Number(summary.measured),
      predicted: Number(summary.predicted),
      missing: Number(summary.missing)
    }
  }

  destroy(): void {
    this.coordinator.clear()
  }
}
