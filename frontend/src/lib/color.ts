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

export const SELECTED = 'oklch(0.637 0.237 25.331)'

export interface Ink {
  text: string
  muted: string
  rule: string
}

export function chartInk(scheme: Scheme): Ink {
  return {
    light: { text: 'oklch(0.145 0 0)', muted: 'oklch(0.556 0 0)', rule: 'oklch(0.922 0 0)' },
    dark: { text: 'oklch(0.985 0 0)', muted: 'oklch(0.708 0 0)', rule: 'oklch(1 0 0 / 10%)' }
  }[scheme]
}
