import { rgb } from 'd3-color'
import { interpolateRgb } from 'd3-interpolate'
import { scaleSequential } from 'd3-scale'
import { interpolateViridis } from 'd3-scale-chromatic'

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

export function falseColors(rgb: Uint8Array): (index: number) => RGB {
  return (index) => [rgb[3 * index], rgb[3 * index + 1], rgb[3 * index + 2]]
}

export const NO_VALUE = { light: '#52525b', dark: '#d4d4d8' }

const BACKGROUND = { light: '#ffffff', dark: '#18181b' }

export const BINS = 32

export const SHADES: string[] = Array.from({ length: BINS }, (_, bin) =>
  interpolateViridis((bin + 0.5) / BINS)
)

function fade(color: string, scheme: Scheme): string {
  return rgb(interpolateRgb(color, BACKGROUND[scheme])(0.6)).formatHex()
}

export function fadedShades(scheme: Scheme): string[] {
  return SHADES.map((color) => fade(color, scheme))
}

export function propertyColors(scheme: Scheme): string[] {
  return [...SHADES, ...fadedShades(scheme), NO_VALUE[scheme]]
}

export const SELECTED = 'oklch(0.637 0.237 25.331)'

export const OUTLINE = [
  { stroke: 'black', width: 4 },
  { stroke: SELECTED, width: 2 }
]

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
