# Testing and benchmarking

Tests run against a synthetic artifact tree built by `scripts/fixture.py`; they need neither Modal nor a GPU, and nothing in them may touch the network.

## Tests

```sh
uv run pytest
uv run pytest -k search  # one area
```

`tests/conftest.py` builds the tree once per session. CI runs the same two ruff
commands, the second as `--check`.

## Benchmarks

```sh
uv run modal run -m scripts.benchmark
uv run modal run -m scripts.benchmark --runs 50
```

The benchmark measures the production artifacts on the Modal volume, never the fixture. `modal run` starts an ephemeral copy of `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency. A client in a separate container times one cold `/meta`, then each endpoint warm, and a container with the server's spec times the stages of `search()`. Latency includes Modal's ingress but not the network of whoever started the run.

Output is JSON recording the commit, dataset revision, date, the Modal spec of server and client, and the thread configuration. Run it only when asked; it never runs in CI. If the code under test needs artifacts the volume does not hold yet, build them first.

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
