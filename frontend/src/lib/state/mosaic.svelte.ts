import { type Projection, downloadUrl, projectionUrl } from '$lib/client'
import { BINS } from '$lib/color'
import { PREDICTABLE, type Property, UNLABELLED } from '$lib/labels'
import { Coordinator, Selection, wasmConnector } from '@uwdata/mosaic-core'
import { loadParquet } from '@uwdata/mosaic-sql'

export interface Coloured {
  table: string
  property: Property
  faded: boolean
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

  #loadProperties(): Promise<unknown> {
    this.#properties ??= this.connect().exec(
      loadParquet('properties', downloadUrl('properties'), { replace: true })
    )
    return this.#properties
  }

  async colour(points: string, property: Property, everywhere: boolean): Promise<Coloured> {
    await this.#loadProperties()
    const database = this.connect()
    const predictable = PREDICTABLE.includes(property)
    const measured = predictable ? `${property}_measured` : property
    const prediction = predictable ? `${property}_predicted` : 'NULL'
    const width = predictable ? `${property}_width` : 'NULL'
    const predicted = everywhere ? 'true' : `${measured} IS NULL`
    const value = everywhere ? prediction : `coalesce(${measured}, ${prediction})`
    const shown = `SELECT galaxy, ${value} AS value, ${predicted} AS predicted,
      CASE WHEN ${predicted} THEN ${width} END AS width FROM properties`
    const summary = (
      await database.query(`
        SELECT quantile_cont(value, 0.01) AS low, quantile_cont(value, 0.99) AS high,
          count(value) FILTER (WHERE NOT predicted) AS measured,
          count(value) FILTER (WHERE predicted) AS predicted,
          count(*) - count(value) AS missing
        FROM (${shown})`)
    ).get(0)
    const low = Number(summary.low)
    const high = Number(summary.high)
    const span = high > low ? high - low : 1
    const faded = predictable && !everywhere
    const values = `${property}_${everywhere ? 'predicted' : 'measured'}`
    await database.exec(`
      CREATE OR REPLACE TABLE ${values} AS
      SELECT *, coalesce(
        least(${BINS - 1}, greatest(0, floor((value - ${low}) / ${span} * ${BINS})))
          + ${faded ? `CASE WHEN predicted THEN ${BINS} ELSE 0 END` : 0},
        ${2 * BINS}
      )::UTINYINT AS colour
      FROM (${shown})`)
    const table = `${points}_${values}`
    await database.exec(`
      CREATE OR REPLACE VIEW ${table} AS
      SELECT p.galaxy, p.x, p.y, p.category, c.value, c.width, c.predicted, c.colour
      FROM ${points} p LEFT JOIN ${values} c USING (galaxy)`)
    return {
      table,
      property,
      faded,
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
