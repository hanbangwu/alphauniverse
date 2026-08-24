export function must<T>(value: T | null | undefined, describe: string): T {
  if (value === null || value === undefined) throw new Error(describe)
  return value
}
