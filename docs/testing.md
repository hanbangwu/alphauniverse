# Testing and benchmarking

Tests run against a synthetic artifact tree built by `scripts/fixture.py`; they need neither Modal, a GPU nor the network.

The tree has the production schemas at a size that fits in a CI runner, so the API and the search path run real code against real files:

- Embeddings are drawn around a fixed set of random cluster centres, not as uniform noise. In 768 dimensions uniform random vectors are all nearly orthogonal, which would make every ranking arbitrary and every recall figure meaningless.
- The 2-d points come from a fixed random projection, standing in for the trained parametric UMAP, which would pull torch and a training run into the tests. One projection serves both point sets, as one trained projector does in production.
- The tree leaves out `codebook` and `parametric_umap`: nothing served reads them, and their absence exercises the 404 path.

## Tests

```sh
uv run pytest
uv run pytest -k search  # one area
uv run ruff check app scripts tests modal_app.py
uv run ruff format app scripts tests modal_app.py
```

`tests/conftest.py` builds the tree once per session.

## Benchmarks

```sh
uv run modal run -m scripts.benchmark
uv run modal run -m scripts.benchmark --runs 50
```

The benchmark measures the production artifacts on the Modal volume, never the fixture. `modal run` starts an ephemeral copy of `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency. A client in a separate container times one cold `/meta` and one cold `/similarity`, then, warm, `/meta`, the image, tokens and coverage endpoints, and patch-only `/similarity` at three values of `matches`. The spectrum endpoints, span queries and artifact downloads are not timed. A container with the server's spec times the startup loads, then the stages of `search()` for the first query and warm.

Output is JSON recording the commit, dataset revision, date, the Modal spec of server and client, and the thread configuration. It never runs in CI.

Read `docs/performance.md` before drawing a conclusion from a benchmark run.

## Frontend

```sh
cd frontend
bun install
bun run lint     # prettier + eslint
bun run check    # regenerate the client from openapi.json, then svelte-check
```

There is no frontend test suite.

## CI

`.github/workflows/ci-cd.yml` runs on pull requests and on pushes to `main`:

1. **python**: `ruff check`, `ruff format --check`, then `uv run pytest`
2. **openapi**: regenerates `frontend/openapi.json` and fails if it differs from the committed copy
3. **frontend**: `bun run lint` and `bun run check`
4. **deploy**: Modal, on pushes to `main` only, and only if the other three pass
