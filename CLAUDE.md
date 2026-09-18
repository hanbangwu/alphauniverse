# CLAUDE.md

alphaUniverse wraps [AION](https://huggingface.co/polymathic-ai/aion-base), Polymathic's
astronomy foundation model, in a web interface for exploring its embeddings: a 2-d
projection of every galaxy, and patch-level similarity search across the survey.

## Layout

| Path             | What it is                                                       |
| ---------------- | ---------------------------------------------------------------- |
| `app/`           | Build pipeline and the FastAPI serving app                        |
| `modal_app.py`   | Modal deployment: three build jobs, one served ASGI app           |
| `scripts/`       | Dataset build, OpenAPI export, test fixtures, benchmarks          |
| `tests/`         | pytest suite, run against a synthetic artifact tree               |
| `frontend/`      | SvelteKit single-page app                                         |
| `docs/`          | Architecture, pipeline, performance and testing notes             |

## Docs

- `docs/architecture.md` — how the pieces fit and where a request's time goes
- `docs/pipeline.md` — the three build stages and the artifact schemas
- `docs/performance.md` — the cost model, measured numbers and known ceilings
- `docs/testing.md` — running tests and benchmarks

## Commands

```sh
uv run pytest                                       # test suite (~6s, no network)
uv run python -m scripts.fixture --galaxies 12      # build a fixture tree
uv run python -m scripts.benchmark --galaxies 32    # measure the search path
uv run python -m scripts.openapi                    # regenerate frontend/openapi.json

cd frontend
bun install && bun run dev                          # needs PUBLIC_API_URL, see .env.example
bun run check                                       # regenerate client + typecheck
bun run lint                                        # prettier + eslint
```

## Invariants

These are load-bearing and nothing asserts most of them at runtime. Breaking one
produces wrong results rather than an error.

- **faiss ids encode position**: `id = galaxy * N_PATCHES + patch`. Holds only
  because `generate_index` adds every galaxy's anchor patches in row order,
  contiguously. `tests/test_search.py` checks it.
- **Anchor coverage is total**: every galaxy has a Legacy Survey match, so the
  anchor column is never null. The other three surveys are nullable.
- **Cell layout**: within a survey's cell, image or spectrum tokens come first,
  then that survey's scalars. `N_PATCHES` slicing depends on it.
- **Artifacts are immutable per revision**: every response is a pure function of
  `DATASET_REVISION` and the files on disk.
- **`GalaxyIndex` is hardcoded** to the production galaxy count and is *not*
  derived from the artifacts. See the strict xfail in `tests/test_api.py`.

## Conventions

- Python: no `from __future__` needed except where `TYPE_CHECKING` guards imports.
  Heavy imports (`torch`, `wandb`) stay inside functions or `TYPE_CHECKING`.
- `app/config.py` holds everything shared between the pipeline and the server.
  `build_dir()` reads `ALPHAUNIVERSE_CACHE` on every call, so a process can be
  pointed at a fixture tree at any time.
- `app/parametric_umap.py` and `app/encode.py` import torch at module level and
  are only installable via the `build` dependency group. Nothing served may
  import them.
- Frontend: Svelte 5 runes throughout. State lives in classes under
  `src/lib/state/`, reached through `getApp()` and friends, not stores.
  `src/lib/api/` is generated from `openapi.json` and gitignored.
- Docstrings describe what the code does now. Leave history to git.

## Gotchas

- `frontend/openapi.json` is committed but generated. Changing a route, a model
  or a route docstring in `app/main.py` changes it; CI fails if it is stale.
- The frontend will not build without `PUBLIC_API_URL`.
- Tests never touch the network. `/galaxies/{g}/image.png` reads the Hugging Face
  dataset, so its test stubs `app.main.image`.
