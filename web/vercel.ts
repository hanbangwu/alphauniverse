import { type VercelConfig, routes } from '@vercel/config/v1'

export const config: VercelConfig = {
  buildCommand:
    'LC_ALL=C.UTF-8 uv run --directory ../docs --group docs sphinx-build -b html . ../web/static/docs && bun run build',
  redirects: [routes.redirect('/docs', '/docs/', { permanent: true })],
  rewrites: [routes.rewrite('/api/(.*)', `${process.env.PUBLIC_API_URL}/$1`)]
}
