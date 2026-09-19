# Testing and benchmarking

Both run against a synthetic artifact tree built by `scripts/fixture.py`; neither needs Modal or a GPU, and nothing in either may touch the network.

## Tests

```sh
uv run pytest
uv run pytest -k search  # one area
```

`tests/conftest.py` builds the tree once per session.

## Benchmarks

```sh
uv run python -m scripts.benchmark --galaxies 32 --out bench.json
uv run python -m scripts.benchmark --tree .cache/fixture   # reuse a tree
```

Output is JSON tagged with the commit and the fixture shape. **Runs only compare when `galaxies` and `seed` match**: the index geometry depends on the dataset size, so a bigger fixture is a different experiment, not a longer one. Load timings run against a warm page cache on a local disk, so they are a lower bound on a Modal container reading a cold network volume.

Before and after a change worth measuring:

```sh
uv run python -m scripts.benchmark --galaxies 32 --out before.json
# ... change something ...
uv run python -m scripts.benchmark --galaxies 32 --out after.json
```

Read `docs/performance.md` before drawing a conclusion from a benchmark run, particularly the section on why fixture recall numbers do not transfer to production.

## Frontend

```sh
cd frontend
bun install
bun run lint     # prettier + eslint
bun run check    # regenerate the client from openapi.json, then svelte-check
```

`bun run check` needs `PUBLIC_API_URL` set; copy `.env.example` to `.env`. There is no frontend test suite.

## CI

`.github/workflows/ci-cd.yml` runs on pull requests and on pushes to `main`:

1. **python**: `uv run pytest`
2. **openapi**: regenerates `frontend/openapi.json` and fails if it differs from the committed copy
3. **frontend**: `bun run lint` and `bun run check`
4. **deploy**: Modal, on pushes to `main` only, and only if the other three pass
