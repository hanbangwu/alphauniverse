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

Two tests in `tests/test_search.py` check what the shared tree cannot show on its own:

- `test_approximate_ranking_agrees_with_exact` compares `search()` with a brute-force ranking over every token. The brute force follows `search()` except for the candidate step, scores the float32 embeddings rather than the index's fp16 copies, and holds the whole corpus in memory, so it only runs at fixture scale. The fixture is small enough that the candidate pool covers it, so the two rankings agree exactly; on production data they would not.
- `test_ids_stay_contiguous_across_add_batches` builds its own index with one galaxy per `add()` call, including galaxies without a spectrum, because the shared tree is smaller than `BATCH` and goes in with a single call. A reordered or dropped batch would shift every galaxy id in every `/similarity` response without an error.

## Benchmarks

```sh
uv run modal run -m scripts.benchmark
uv run modal run -m scripts.benchmark --runs 50
```

The benchmark measures the production artifacts on the Modal volume, never the fixture. `modal run` starts an ephemeral copy of `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency. A client in a separate container times one cold `/meta` and one cold `/similarity`, then, warm, `/meta`, the image, tokens and coverage endpoints, and patch-only `/similarity` at three values of `matches`. The spectrum endpoints, span queries and artifact downloads are not timed. A container with the server's spec times the startup loads, then the stages of `search()` for the first query and warm, for two versions of the code on the same host: the head of `origin/main` (before) and the checked-out change (after). Each version is sent as a `git archive` of `app/`, `scripts/` and `modal_app.py`, and runs its own `stages` in a subprocess, in the order before, after, after, before. All versions use the change's locked dependencies. A version that fails, for example because it cannot read the current artifacts, is reported with its error. Request latency is timed for the change only.

Output is JSON recording the commit, dataset revision, date, the Modal spec of server and client, the thread configuration, and the CPU model of the client and stage containers. It never runs in CI.

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

`.github/workflows/ci-cd.yml` runs on pull requests and on pushes to `main`, except a push that changes only `CLAUDE.md`, `.claude/rules/`, `README.md` or `docs/`, whose pull request already ran the checks:

1. **python**: `ruff check`, `ruff format --check`, then `uv run pytest`
2. **openapi**: regenerates `frontend/openapi.json` and fails if it differs from the committed copy
3. **frontend**: `bun run lint` and `bun run check`
4. **deploy**: Modal, on pushes to `main` only, and only if the other three pass

Every uv command in the workflow runs with `UV_LOCKED=1`, so a `pyproject.toml` change without a matching `uv.lock` fails CI instead of being re-resolved.

Lint fails on a comment that starts with `TODO`, `FIXME`, `HACK` or `XXX`, in any case: ruff's `FIX` rules check Python comments, and eslint's `no-warning-comments` checks JavaScript, TypeScript and Svelte `<script>` comments, after any leading `*`. Comments in other files and in Svelte markup or styles are not checked.

`.github/workflows/rules.yml` runs on pull requests. Its Rules job fails on a commit in the pull request authored by `noreply@anthropic.com`, a GitHub `[bot]` account or Copilot, and on one committed by `noreply@anthropic.com` without a `Co-Authored-By` trailer naming that address.
