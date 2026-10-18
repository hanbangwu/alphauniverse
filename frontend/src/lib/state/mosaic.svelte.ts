import { type Projection, downloadUrl, projectionUrl } from '$lib/client'
import { BINS } from '$lib/color'
import { PREDICTABLE, type Property } from '$lib/labels'
import { Coordinator, Selection, wasmConnector } from '@uwdata/mosaic-core'
import { Query, column, count, isBetween, isIn, loadParquet } from '@uwdata/mosaic-sql'
import type { Point, Rectangle } from 'embedding-atlas'

function bounds(polygon: Point[]): Rectangle {
  const xs = polygon.map(({ x }) => x)
  const ys = polygon.map(({ y }) => y)
  return {
    xMin: Math.min(...xs),
    xMax: Math.max(...xs),
    yMin: Math.min(...ys),
    yMax: Math.max(...ys)
  }
}

function inside(x: number, y: number, polygon: Point[]): boolean {
  let crossed = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i]
    const b = polygon[j]
    if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) crossed = !crossed
  }
  return crossed
}

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
    await database.exec(loadParquet(table, projectionUrl(projection), { replace: true }))
    database.clear({ clients: false, cache: true })
    return table
  }

  loadProperties(): Promise<unknown> {
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
    await this.loadProperties()
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
      SELECT p.galaxy, p.x, p.y, c.value, c.width, c.predicted, c.colour
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

  async inRegion(
    table: string,
    region: Rectangle | Point[],
    range: readonly [number, number] | null
  ): Promise<number[]> {
    const box = 'xMin' in region ? region : bounds(region)
    const where = [
      isBetween(column('x'), [box.xMin, box.xMax]),
      isBetween(column('y'), [box.yMin, box.yMax]),
      ...(range ? [isBetween(column('value'), [...range])] : [])
    ]
    const rows = await this.connect().query(
      Query.from(table).select('galaxy', 'x', 'y').where(where)
    )
    const galaxies = Array.from(rows)
      .filter(({ x, y }) => 'xMin' in region || inside(x, y, region))
      .map(({ galaxy }) => Number(galaxy))
      .sort((a, b) => a - b)
    return galaxies.filter((galaxy, index) => galaxy !== galaxies[index - 1])
  }

  async csv(galaxies: number[]): Promise<string> {
    await this.loadProperties()
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
