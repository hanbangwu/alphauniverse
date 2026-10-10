export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'object' && error !== null && 'detail' in error) {
    const { detail } = error
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail.map((entry: { msg: string }) => entry.msg).join('; ')
    }
  }
  return typeof error === 'string' && error ? error : 'The request failed.'
}
