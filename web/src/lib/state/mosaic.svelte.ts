import type { GetProjectionsData, GetPropertiesData } from '$lib/api'
import type { GalaxyId } from '$lib/client'
import { type Projection, client } from '$lib/client'
import { BINS } from '$lib/color'
import { extent } from '$lib/data/scores'
import type { Property } from '$lib/labels'
import { Coordinator, Selection, wasmConnector } from '@uwdata/mosaic-core'
import { Query, column, isBetween, loadParquet } from '@uwdata/mosaic-sql'
import { polygonContains } from 'd3-polygon'
import type { Point, Rectangle } from 'embedding-atlas'
import { SvelteSet } from 'svelte/reactivity'

function bounds(polygon: Point[]): Rectangle {
  const [xMin, xMax] = extent(polygon.map(({ x }) => x))
  const [yMin, yMax] = extent(polygon.map(({ y }) => y))
  return { xMin, xMax, yMin, yMax }
}

export interface Values {
  name: string
  property: Property
  low: number
  high: number
  least: number
  most: number
}

export interface Colored extends Values {
  table: string
}

export class MosaicState {
  readonly coordinator = new Coordinator()
  readonly filter = Selection.intersect()

  readonly predictable = new SvelteSet<string>()

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
      loadParquet(
        table,
        client.buildUrl<GetProjectionsData>({
          url: '/projections/{projection}',
          path: { projection }
        }),
        { replace: true }
      )
    )
    database.clear({ clients: false, cache: true })
    return table
  }

  loadProperties(): Promise<unknown> {
    this.#properties ??= this.connect()
      .exec(
        loadParquet(
          'properties',
          client.buildUrl<GetPropertiesData>({ url: '/galaxies/property' }),
          { replace: true }
        )
      )
      .then(() => this.#readPredictable())
      .catch((error: unknown) => {
        this.#properties = null
        throw error
      })
    return this.#properties
  }

  async #readPredictable(): Promise<void> {
    const columns = await this.connect().query(`SELECT column_name FROM (DESCRIBE properties)`)
    for (const { column_name } of columns) {
      const name = String(column_name)
      if (name.endsWith('_predicted')) this.predictable.add(name.slice(0, -'_predicted'.length))
    }
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
    const predictable = this.predictable.has(property)
    const measured = predictable ? `${property}_measured` : property
    const prediction = predictable ? `${property}_predicted` : 'NULL'
    const width = predictable ? `${property}_width` : 'NULL'
    const predicted = everywhere ? 'true' : `${measured} IS NULL`
    const value = everywhere ? prediction : `coalesce(${measured}, ${prediction})`
    const shown = `SELECT object_id, ${value} AS value, ${predicted} AS predicted,
      CASE WHEN ${predicted} THEN ${width} END AS width FROM properties`
    const summary = (
      await database.query(`
        SELECT quantile_cont(value, 0.01) AS low, quantile_cont(value, 0.99) AS high,
          min(value) AS least, max(value) AS most
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
      )::UTINYINT AS color
      FROM (${shown})`)
    return {
      name,
      property,
      low,
      high,
      least: Number(summary.least),
      most: Number(summary.most)
    }
  }

  async color(points: string, property: Property, everywhere: boolean): Promise<Colored> {
    const values = await this.#values(property, everywhere)
    const table = `${points}_${values.name}`
    await this.connect().exec(`
      CREATE OR REPLACE VIEW ${table} AS
      SELECT p.object_id, p.x, p.y, c.value, c.width, c.predicted, c.color
      FROM ${points} p LEFT JOIN ${values.name} c USING (object_id)`)
    return { ...values, table }
  }

  async inRegion(
    table: string,
    region: Rectangle | Point[],
    range: readonly [number, number] | null
  ): Promise<GalaxyId[]> {
    const box = 'xMin' in region ? region : bounds(region)
    const where = [
      isBetween(column('x'), [box.xMin, box.xMax]),
      isBetween(column('y'), [box.yMin, box.yMax]),
      ...(range ? [isBetween(column('value'), [...range])] : [])
    ]
    const rows = await this.connect().query(
      Query.from(table).select('object_id', 'x', 'y').where(where).orderby('object_id')
    )
    const corners = 'xMin' in region ? null : region.map(({ x, y }): [number, number] => [x, y])
    return Array.from(rows)
      .filter(({ x, y }) => !corners || polygonContains(corners, [x, y]))
      .map(({ object_id }) => String(object_id))
  }

  destroy(): void {
    this.coordinator.clear()
  }
}
