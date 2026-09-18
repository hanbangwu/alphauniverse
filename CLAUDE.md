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

Read **Simplicity** and **Writing** below before adding to any of them.

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

## Workflow

- **Never commit or push to `main`.** Work on a branch, always.
- **Every change lands as a pull request**, with a review requested from Copilot.
  No direct pushes to `main`, however small the change.
- `main` deploys to Modal on merge, and CI gates it on tests, lint, typecheck and
  the OpenAPI drift check. A red PR is not ready.

## Invariants

Code depends on each of these, and nothing asserts most of them at runtime.
Breaking one produces wrong results rather than an error.

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

## Simplicity

Code, tests and documentation are held to one standard: as simple and as small
as the job allows. This is the constraint that governs the others.

- Prefer removing to adding. A change that deletes a concept is usually better
  than one that adds a flag.
- Documentation states what the code does now. Not what it used to do, not what
  it might do, not which alternatives were rejected. Git holds the history.
- Each fact has one home. If something is documented in two places, one of them
  is already wrong.
- When adding to a doc, re-read the whole file first and cut what the addition
  duplicates. Otherwise they accumulate duplicates.
- Working notes stay out of the repository. Scratch analysis, session logs and
  investigation write-ups are thrown away, not committed.
- Tests assert the behaviour that matters, not every property that happens to
  be observable.

### Where future work is recorded

No `TODO`, `FIXME`, `HACK` or `XXX` comments anywhere in the codebase. An inline
marker is tracked by nothing and goes stale without anyone noticing. Known work
lives in exactly three places:

1. **Performance work** — `Candidate work` in `docs/performance.md`, ordered by
   measured impact and re-ordered when a measurement changes it. Entries are
   named, not numbered, so references to them survive a re-ordering.
2. **Behaviour known to be wrong** — a test marked `xfail(strict=True)`, so the
   fix cannot land without someone removing the marker. See `tests/test_api.py`.
3. **Everything else** — a GitHub issue.

## Writing

Two audiences: humans read `README.md`, `docs/`, docstrings, PR descriptions and
review replies; agents read this file and anything written to orient a future
session. The same standard applies to both.

### Say what you mean

Mannered prose substitutes metaphor and flourish for direct statement: "a dial
worth turning" instead of "a parameter worth varying", "this point earns its
keep" instead of "this point still matters". The phrases display the writer
rather than convey the idea, and readers can tell. They are also imprecise: a
metaphor carries connotations the writer did not choose and cannot control. When a literal phrase is available, use it.

### Formatting

- Use lists and headings when asked for them, or when the content is
  multifaceted enough that they aid clarity. Not by default.
- If minimal formatting is requested, use none: no bullets, headings, lists or
  bold.
- Keep conversational or personal exchanges in plain prose.

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

## Performance shape

Measured against production; `docs/performance.md` has the detail. Worth knowing
before touching anything in the serving path:

- **Cold start is ~47 s** — a 15.5 GB index read with no mmap, one container,
  five-minute scaledown. SSR blocks on `/meta`, so the page is blank throughout.
- **Cutouts are re-encoded per request**, ~350 ms each with no useful
  concurrency, so a 32-row match list takes ~11 s.
- **`/similarity` costs scale with `matches`, not patch count** — ~100 ms at the
  default 32, ~325 ms at 128. 90% of that is reconstructing candidate vectors
  through the IVF direct map; the ANN search itself is 3%.
- **No endpoint sets `Cache-Control`**, though every response is immutable per
  revision.

## Gotchas

- `frontend/openapi.json` is committed but generated. Changing a route, a model
  or a route docstring in `app/main.py` changes it; CI fails if it is stale.
- The frontend will not build without `PUBLIC_API_URL`.
- Tests never touch the network. `/galaxies/{g}/image.png` reads the Hugging Face
  dataset, so its test stubs `app.main.image`.
