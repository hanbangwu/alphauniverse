# Testing and benchmarking

Tests run against a synthetic artifact tree built by `scripts/fixture.py`; they need neither Modal, a GPU nor the network.

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

The benchmark measures the production artifacts on the Modal volume, never the fixture. `modal run` starts an ephemeral copy of `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency. A client in a separate container times one cold `/meta`, then each endpoint warm, and a container with the server's spec times the stages of `search()`.

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
