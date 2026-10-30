import type { Meta, TableRow } from '$lib/api'
import { ANCHOR, IMAGE_MODES, SPECTRUM_MODES, surveyOf } from '$lib/data/modes'
import { row } from '$lib/data/scores'
import { type SimilarityQuery, type SimilarityResult, isModeQuery } from '$lib/data/similarity'
import { spectrumTokenOf } from '$lib/data/spectra'
import type { Ecsv, EcsvColumn } from '$lib/download'

interface Identity {
  object_id: string
  ra: number
  dec: number
}

interface TableValue {
  name: string
  survey: string
}

function chunks(values: Float32Array, width: number): number[][] {
  return Array.from({ length: values.length / width }, (_, at) =>
    Array.from(values.subarray(at * width, (at + 1) * width))
  )
}

function tableValueNames(rows: TableRow[], count: number): TableValue[] {
  const names: TableValue[] = Array.from({ length: count })
  names[0] = { name: 'redshift', survey: 'redshift' }
  for (const { section, column, table_value } of rows) {
    if (section !== 'redshift' && table_value !== null)
      names[table_value] = { name: column, survey: section }
  }
  return names
}

function selection(request: SimilarityQuery, meta: Meta, names: TableValue[]) {
  return {
    ...Object.fromEntries(
      IMAGE_MODES.filter((mode) => request[mode]?.length).map((mode) => [
        mode,
        request[mode]!.map((slot) => [Math.floor(slot / meta.grid), slot % meta.grid])
      ])
    ),
    ...Object.fromEntries(
      SPECTRUM_MODES.filter((mode) => request[mode]?.length).map((mode) => [
        mode,
        request[mode]!.map((slot) => spectrumTokenOf(meta, slot))
      ])
    ),
    ...(request.table_values?.length && {
      table_values: request.table_values.map((slot) => names[slot].name)
    })
  }
}

export function searchEcsv(
  result: SimilarityResult,
  meta: Meta,
  identities: Identity[],
  tableRows: TableRow[],
  url: string
): Ecsv {
  const whole = isModeQuery(result.request)
  const score = whole ? 'score' : 'similarity'
  const count = result.galaxies.length
  const width = result.tableValues.length / count
  const names = tableValueNames(tableRows, width)
  const observes = (survey: string, index: number) =>
    survey === ANCHOR || (result.observed[survey]?.[index] ?? false)
  const mapAt = (map: Float32Array, index: number) => row(map, index, map.length / count)

  const columns: EcsvColumn[] = [
    { name: 'object_id', datatype: 'string' },
    { name: 'ra', datatype: 'float64', unit: 'deg' },
    { name: 'dec', datatype: 'float64', unit: 'deg' },
    { name: score, datatype: 'float32' },
    ...IMAGE_MODES.map((mode) => ({
      name: mode,
      datatype: 'string' as const,
      subtype: `float32[${meta.grid},${meta.grid}]`
    })),
    ...SPECTRUM_MODES.map((mode) => ({
      name: mode,
      datatype: 'string' as const,
      subtype: `float32[${result.maps[mode].length / count}]`
    })),
    ...names.map(({ name }) => ({ name, datatype: 'float32' as const }))
  ]

  const rows = identities.map((identity, index) => ({
    ...identity,
    [score]: result.scores[index],
    ...Object.fromEntries(
      IMAGE_MODES.map((mode) => [
        mode,
        observes(surveyOf(mode), index)
          ? JSON.stringify(chunks(mapAt(result.maps[mode], index), meta.grid))
          : null
      ])
    ),
    ...Object.fromEntries(
      SPECTRUM_MODES.map((mode) => [
        mode,
        observes(surveyOf(mode), index)
          ? JSON.stringify(Array.from(mapAt(result.maps[mode], index)))
          : null
      ])
    ),
    ...Object.fromEntries(
      names.map(({ name, survey }, slot) => [
        name,
        observes(survey, index) ? mapAt(result.tableValues, index)[slot] : null
      ])
    )
  }))

  return {
    columns,
    rows,
    meta: {
      rows: 'The query galaxy, then its matches in rank order',
      search: whole ? 'whole modes' : 'tokens',
      request: result.request,
      ...(!isModeQuery(result.request) && {
        selection: selection(result.request, meta, names)
      }),
      url
    }
  }
}
