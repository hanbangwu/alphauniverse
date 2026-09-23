---
paths:
  - "frontend/**"
---

# Frontend

- Never edit `src/lib/components/ui`. It is changed only by `bun run shadcn` and formatted by `bun run format`.
- `src/lib/api` is generated from `openapi.json`, and `openapi.json` from `app/main.py`. When needed, regenerate through `uv run python -m scripts.openapi` then `bun run check`.
- Components never call the API. Each call goes through the generated SDK in `$lib/api`, wrapped as TanStack Query options in `src/lib/data/queries.ts`.
- App-wide state is a class in `src/lib/state/*.svelte.ts`, reached through the getters in `app.svelte.ts`. State shared by one group of components lives beside them as a `*.svelte.ts` class with its own `Context`, as `components/similarity/similarity.svelte.ts` does.
