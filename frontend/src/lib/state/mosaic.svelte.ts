import { type Projection, downloadUrl, projectionUrl } from '$lib/client'
import { BINS } from '$lib/color'
import { PREDICTABLE, type Property, UNLABELLED } from '$lib/labels'
import { Coordinator, Selection, wasmConnector } from '@uwdata/mosaic-core'
import { Query, column, count, isBetween, isIn, loadParquet } from '@uwdata/mosaic-sql'

export interface Values {
  name: string
  property: Property
  faded: boolean
  low: number
  high: number
  least: number
  most: number
  measured: number
  predicted: number
  missing: number
}

export interface Coloured extends Values {
  table: string
}

export class MosaicState {
  readonly coordinator = new Coordinator()
  readonly filter = Selection.intersect()
  readonly colouredFilter = Selection.intersect({ include: this.filter })
  readonly region = Selection.intersect()

  #connected = false
  #properties: Promise<unknown> | null = null
  readonly #tables: Record<string, Promise<Values>> = {}

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
    this.#properties ??= this.connect()
      .exec(loadParquet('properties', downloadUrl('properties'), { replace: true }))
      .catch((error: unknown) => {
        this.#properties = null
        throw error
      })
    return this.#properties
  }

  #values(property: Property, everywhere: boolean): Promise<Values> {
    const name = `${property}_${everywhere ? 'predicted' : 'measured'}`
    this.#tables[name] ??= this.#build(name, property, everywhere).catch((error: unknown) => {
      delete this.#tables[name]
      throw error
    })
    return this.#tables[name]
  }

  async #build(name: string, property: Property, everywhere: boolean): Promise<Values> {
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
          min(value) AS least, max(value) AS most,
          count(value) FILTER (WHERE NOT predicted) AS measured,
          count(value) FILTER (WHERE predicted) AS predicted,
          count(*) - count(value) AS missing
        FROM (${shown})`)
    ).get(0)
    const low = Number(summary.low)
    const high = Number(summary.high)
    const span = high > low ? high - low : 1
    const faded = predictable && !everywhere
    await database.exec(`
      CREATE OR REPLACE TABLE ${name} AS
      SELECT *, coalesce(
        least(${BINS - 1}, greatest(0, floor((value - ${low}) / ${span} * ${BINS})))
          + ${faded ? `CASE WHEN predicted THEN ${BINS} ELSE 0 END` : 0},
        ${2 * BINS}
      )::UTINYINT AS colour
      FROM (${shown})`)
    return {
      name,
      property,
      faded,
      low,
      high,
      least: Number(summary.least),
      most: Number(summary.most),
      measured: Number(summary.measured),
      predicted: Number(summary.predicted),
      missing: Number(summary.missing)
    }
  }

  async colour(points: string, property: Property, everywhere: boolean): Promise<Coloured> {
    const values = await this.#values(property, everywhere)
    const table = `${points}_${values.name}`
    await this.connect().exec(`
      CREATE OR REPLACE VIEW ${table} AS
      SELECT p.galaxy, p.x, p.y, p.category, c.value, c.width, c.predicted, c.colour
      FROM ${points} p LEFT JOIN ${values.name} c USING (galaxy)`)
    return { ...values, table }
  }

  async within(name: string, [low, high]: readonly [number, number]): Promise<number> {
    const counted = await this.connect().query(
      Query.from(name)
        .select({ count: count() })
        .where(isBetween(column('value'), [low, high]))
    )
    return Number(counted.get(0).count)
  }

  async inRegion(table: string, filter: Selection): Promise<number[]> {
    const rows = await this.connect().query(
      Query.from(table)
        .select('galaxy')
        .distinct()
        .where([this.region, filter].flatMap((selection) => selection.predicate(null) ?? []))
        .orderby('galaxy')
    )
    return Array.from(rows, ({ galaxy }) => Number(galaxy))
  }

  async csv(galaxies: number[]): Promise<string> {
    await this.#loadProperties()
    const rows = await this.connect().query(
      Query.from('properties')
        .select('*')
        .where(isIn(column('galaxy'), galaxies))
        .orderby('galaxy')
    )
    const names = rows.schema.fields.map(({ name }: { name: string }) => name)
    const lines = Array.from(rows, (row) => names.map((name: string) => row[name] ?? '').join(','))
    return [names.join(','), ...lines].join('\n')
  }

  destroy(): void {
    this.coordinator.clear()
  }
}
