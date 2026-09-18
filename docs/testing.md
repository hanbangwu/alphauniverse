# Testing and benchmarking

Both run against a synthetic artifact tree, so neither needs Modal, a GPU or the
network.

## Tests

```sh
uv run pytest            # ~6s
uv run pytest -k search  # one area
```

`tests/conftest.py` builds one tree per session with `scripts.fixture` and points
the app at it by setting `ALPHAUNIVERSE_CACHE`. This works at any time because
`build_dir()` re-reads the environment on every call; the `@cache`d handles in
`app.search` and `app.main` are cleared whenever the tree changes under them.

- `tests/test_api.py` — the HTTP contract: response shapes, binary encodings,
  validation, and the 404 paths for both an unknown artifact role and a known
  role with no file.
- `tests/test_search.py` — the index layout invariant, the ranking contract, and
  agreement with brute force. `test_ids_stay_contiguous_across_add_batches`
  builds its own small tree with `BATCH` patched down, because the shared
  fixture is smaller than one batch and so cannot exercise contiguity across
  several `add()` calls.
- `tests/test_fixture.py` — properties the fixture itself must hold, chiefly that
  both point sets are coordinates in one projected space, as production's single
  trained projector guarantees and the fixture has to arrange.

Two things to know when adding tests:

- **Nothing may touch the network.** `/galaxies/{g}/image.png` reads the Hugging
  Face dataset, so its test stubs `app.main.image`.
- **The fixture omits `codebook` and `parametric_umap`**, and its `full_points`
  covers the anchor survey only — so it is grouped by galaxy, while production
  streams survey by survey and is not. `test_fixture_groups_full_points_by_galaxy`
  is scoped to the fixture for that reason; nothing may read the real artifact
  expecting a monotonic galaxy column.

### The strict xfail

`test_galaxy_past_the_end_is_rejected` is marked `xfail(strict=True)`. It records
a real gap: `GalaxyIndex` hardcodes the production galaxy count rather than
deriving it from the artifacts, so a row index past the end of the data passes
validation and fails somewhere less helpful. When that is fixed the test will
pass, and `strict=True` turns a pass into a failure — so the fix cannot land
without also removing the marker. That is the intent; do not relax it.

## Benchmarks

```sh
uv run python -m scripts.benchmark --galaxies 32 --out bench.json
uv run python -m scripts.benchmark --tree .cache/fixture   # reuse a tree
```

Measures artifact sizes with a linear projection to production scale, index load
time, query latency across query shapes, a stage-by-stage split of one query, and
recall against brute force swept over `nprobe`.

The galaxy count is read off the tree, not from `--galaxies`, so `--tree` works
against a tree of any size and the flag only matters when building one.

Output is JSON tagged with the commit and the fixture shape. **Runs only compare
when `galaxies` and `seed` match** — the index geometry depends on the dataset
size, so a bigger fixture is a different experiment, not a longer one.

Before and after a change worth measuring:

```sh
uv run python -m scripts.benchmark --galaxies 32 --out before.json
# ... change something ...
uv run python -m scripts.benchmark --galaxies 32 --out after.json
```

Read `docs/performance.md` before drawing a conclusion from a benchmark run —
particularly the section on why fixture recall numbers do not transfer to
production.

### Keeping `measure_phases` honest

`measure_phases` re-implements `app.search.search` step by step so the stages can
be timed separately. It will drift silently if `search` changes and it does not.
Change them together.

## Frontend

```sh
cd frontend
bun install
bun run lint     # prettier + eslint
bun run check    # regenerate the client from openapi.json, then svelte-check
```

`bun run check` needs `PUBLIC_API_URL` set — copy `.env.example` to `.env`. There
is no frontend test suite yet.

## CI

`.github/workflows/ci-cd.yml` runs on pull requests and on pushes to `main`:

1. **python** — `uv run pytest`
2. **frontend** — `bun run lint` and `bun run check`
3. **openapi** — regenerates `frontend/openapi.json` and fails if it differs from
   the committed copy
4. **deploy** — Modal, on pushes to `main` only, and only if all three pass

If the openapi job fails, run `uv run python -m scripts.openapi` and commit the
result. Route docstrings feed the schema, so editing one changes the file.
