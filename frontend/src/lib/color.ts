import { must } from './invariant'
import { rgb } from 'd3-color'
import { scaleSequential } from 'd3-scale'
import { interpolateViridis, schemeObservable10 } from 'd3-scale-chromatic'
import { defaultCategoryColors } from 'embedding-atlas'

export type RGB = [number, number, number]
export type Scheme = 'light' | 'dark'

export function toRGB(value: string): RGB {
  const { r, g, b } = rgb(value)
  return [r, g, b]
}

export function continuous(domain: readonly [number, number]): (value: number) => RGB {
  const [low, high] = domain
  const scale = scaleSequential(interpolateViridis)
    .domain(low === high ? [low, low + 1] : [low, high])
    .clamp(true)
  return (value) => toRGB(scale(value))
}

const TOKEN_PALETTE: RGB[] = schemeObservable10.map(toRGB)

export function tokenColors(values: ArrayLike<number>): (value: number) => RGB {
  const counts = new Map<number, number>()
  const first = new Map<number, number>()
  for (let i = 0; i < values.length; i++) {
    const token = values[i]
    counts.set(token, (counts.get(token) ?? 0) + 1)
    if (!first.has(token)) first.set(token, i)
  }

  const ranks = new Map(
    [...counts.keys()]
      .sort((a, b) => counts.get(b)! - counts.get(a)! || first.get(a)! - first.get(b)!)
      .map((token, rank): [number, number] => [token, rank])
  )

  return (value) =>
    TOKEN_PALETTE[must(ranks.get(value), `rank for token ${value}`) % TOKEN_PALETTE.length]
}

export function morphologyColors(classes: number, scheme: Scheme): string[] {
  return [...defaultCategoryColors(classes), { light: '#52525b', dark: '#d4d4d8' }[scheme]]
}
