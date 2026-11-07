import { type Handle, redirect } from '@sveltejs/kit'

export const handle: Handle = ({ event, resolve }) => {
  if (event.route.id !== '/login' && !event.cookies.get('password')) redirect(307, '/login')
  return resolve(event)
}
