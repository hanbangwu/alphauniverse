# Testing and benchmarking

## Tests

```sh
uv run pytest
uv run pytest -k search  # one area
uv run ruff check app scripts tests modal_app.py
uv run ruff format app scripts tests modal_app.py
```

Tests need no Modal, GPU or network. `tests/conftest.py` builds one synthetic tree per session with `scripts/fixture.py`, in the production schemas:

- Embeddings cluster around fixed random centres. Uniform noise in 768 dimensions is nearly orthogonal, which would make every ranking arbitrary.
- The 2-d points come from one fixed random projection in place of the trained parametric UMAP, so the default suite needs no torch. One projection serves both point sets, as the projector does in production.
- `codebook` and `parametric_umap` are left out: nothing served reads them, and their absence exercises the 404 path.

Four tests in `tests/test_search.py` check what the shared tree cannot show:

- `test_approximate_ranking_agrees_with_exact` compares `search()` with `exact_ranking` in `scripts/recall.py`: the same search without the candidate step, a brute force over every token's float32 embedding in place of the index's fp16 copies. It asks for fewer matches than the corpus holds, so the candidate step must choose. It checks the galaxies, their order, each score and each patch and span map (null where the galaxy has no spectrum) to within `SCORE_TOLERANCE`, after asserting that the exact scores are more than twice that apart. On production data the two need not agree; `scripts/recall.py` measures how often they do.
- `test_ids_stay_contiguous_across_add_batches` builds its own index with one galaxy per `add()` call, galaxies without a spectrum included, since the shared tree goes in with one call. A reordered or dropped batch would silently shift every galaxy id.
- `test_rank_keeps_the_query_first_and_each_row_together` calls `rank()` on arrays built in the test, since on the shared tree `candidates()` already returns galaxies in ranked order.
- `test_a_search_that_finds_too_few_looks_further` changes `PROBE` and `NPROBE` so that the first search falls short, since at their defaults it finds every galaxy on the shared tree. It asks for one vector over one list, 2048 vectors over one list, and one vector over every list, so that the vectors, the lists, or both must widen.

The torch modules' tests skip unless the `build` group is installed:

```sh
uv run --group build pytest tests/test_encode.py tests/test_parametric_umap.py
```

- They build AION and its codecs with random weights from the configs in `tests/aion/`, copied from `polymathic-ai/aion-base`. No test downloads weights.
- `test_saved_weights_load_back_unchanged` round-trips a small AION through `save_pretrained` and `from_pretrained`, which needs `safetensors`, from aion's `torch` extra.
- `tests/test_parametric_umap.py` trains the projector for one epoch on a four-galaxy tree, twice, and checks the results match.
- `test_padded_sdss_spectra_keep_their_flux` is `xfail(strict=True)`: SDSS spectra end in `lambda = -1` padding that zeroes the codec input. Remove the mark once that is fixed.

## Benchmarks

```sh
uv run modal run -m scripts.benchmark
uv run modal run -m scripts.benchmark --runs 50
```

The benchmark measures the production artifacts on the Modal volume. It refuses to run with uncommitted changes, and fetches `origin/main` first. `modal run` starts an ephemeral copy of `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency.

A client in a separate container times, in order:

1. one cold `/meta` and one cold `/search`;
2. warm, `runs` times each: `/meta`, the image, image tokens and galaxy endpoints, and patch-only `/search` at 8, 32 and 128 matches;
3. warm: both spectrum routes, and `/search` at 32 matches with 4 spans, alone and with 4 patches. Galaxies with a DESI spectrum come from one download of `tokens`. Spans are drawn from those the spectrum covers, as in `scripts/recall.py`;
4. concurrent requests at 1, 4, `max_inputs` and twice `max_inputs` clients (16 and 32 today). Each client is a thread with its own connection, sending patch-only `/search` at 32 matches. All clients warm their connections, then each sends `runs` timed requests. Each level reports p50, p95 and requests per second. Past `max_inputs`, requests wait for a slot. The client reserves 1 CPU, so a level where throughput stops rising may be the client's limit.

Request latency is timed for the checked-out commit only. Artifact downloads are not timed.

A container with the server's spec times the startup loads and the stages of `search()`, first query and warm, for up to three versions on one host: the checked-out commit (after), `origin/main` (before), and the stored report's best version unless its code matches one of those. Each version is a `git archive` of `app/`, `scripts/` and `modal_app.py`, running its own `stages` in a subprocess with the checked-out commit's locked dependencies. The order is mirrored: after, before, best, then back. After goes first, so its first round has the cold page cache. A failed round is reported with its error.

Each version's `stages` also times `search()` whole at 8, 32 and 128 matches, on the same patch-only queries, under `searches`. For each it reports p50 and p95, the share of queries that searched the index more than once (`looked_further`), and the most searches one query took, counted by faiss's `indexIVF_stats` on the timed call. These times are separate from `total_p50_ms`, which still picks the best version. A version from before this measurement reports the stages only.

The report goes to `docs/benchmarks/latest.json`, replacing the last one; commit it before the next run. It records the commits, each round's dataset revision, the date, the server's CPU, memory and concurrency settings (`max_inputs`, `max_containers`, `scaledown_window`, from `modal_app.py`), the client's Modal spec, the thread configuration and each container's CPU identity. It names the best version: the lowest `total_p50_ms` (the sum of warm stage medians at 32 matches), averaged over its rounds, among versions whose rounds all succeeded.

- A stored best commit missing from the clone is noted and dropped.
- If the request timings or the stage container fail as a whole, the report records the error and keeps the other part.
- When no version's rounds all succeeded, the report keeps the stored best commit, without its old figure, and notes it.

```sh
uv run modal run -m scripts.recall
uv run modal run -m scripts.recall --per-kind 50
```

`scripts/recall.py` measures how often `search()` returns the galaxies an exact search ranks highest. A container on the build image, with the build jobs' CPU and memory and the volume read-only, loads every patch and span embedding. It then runs `--per-kind` queries (100 by default) of each kind, each kind with its own seed: 4 patches, 4 spans, and both, at 32 matches. Spans are drawn as in the benchmark. A query's recall is the share of `exact_ranking`'s 32 galaxies that `search()`, at the served `PROBE` and `NPROBE`, also returns. Each kind reports its mean, its minimum, the share of queries that found all 32, and the share where `search()` returned fewer than 32. The report adds the commit, dataset revision, date, Modal spec, thread configuration, corpus load time, total time and peak memory. It goes to `docs/benchmarks/recall.json`; commit it before the next run.

### From GitHub Actions

Nothing starts either script automatically. With write access, start the Benchmark workflow from the Actions tab and pick:

- the branch, which must contain the workflow (merge `main` into an older branch first);
- `benchmark` or `recall`;
- optionally `runs`, passed to the benchmark as `--runs`, or `per_kind`, passed to recall as `--per-kind`. The other script's input is ignored, and an empty one keeps the default.

The job runs with full history and the deploy job's Modal token. The report appears in the run's summary and log, and as its artifact, the JSON file itself. Commit it by hand: the Rules job rejects commits by a bot. GitHub stops a job after six hours, which leaves no report; how long a run takes is not yet measured.

Read `docs/performance.md` before drawing a conclusion from a run.

## Frontend

```sh
cd frontend
bun install
bun run lint     # prettier + eslint
bun run check    # regenerate the client from openapi.json, then svelte-check
```

There is no frontend test suite.

## CI

`.github/workflows/ci-cd.yml` runs on pull requests and on pushes to `main`, except pushes that change only `CLAUDE.md`, `.claude/rules/`, `README.md` or `docs/`:

1. **python**: `ruff check`, `ruff format --check`, then `uv run pytest`
2. **openapi**: regenerates `frontend/openapi.json` and fails if it differs from the committed copy
3. **frontend**: `bun run lint` and `bun run check`
4. **deploy**: Modal, on pushes to `main` once the other three pass

Every uv command runs with `UV_LOCKED=1`, so a `pyproject.toml` change without its `uv.lock` fails.

Lint fails on a comment starting with `TODO`, `FIXME`, `HACK` or `XXX`, in any case: ruff's `FIX` rules check Python, and eslint's `no-warning-comments` checks JavaScript, TypeScript and Svelte `<script>`, after any leading `*`. Other files, and Svelte markup and styles, are not checked.

The other workflows:

- `build.yml` runs the torch tests on pull requests that change the torch modules or what they import (`app/encode.py`, `app/parametric_umap.py`, `app/config.py`, `app/dataset.py`, `app/search.py`), what the tests build trees with (`app/images.py`, `app/spectra.py`, `scripts/fixture.py`), `pyproject.toml`, `uv.lock`, those tests, `tests/aion/` or itself.
- `benchmark.yml` runs only when started by hand (above).
- `rules.yml` runs on pull requests. It fails on a commit authored by `noreply@anthropic.com`, a GitHub `[bot]` account or Copilot, and on one committed by `noreply@anthropic.com` without a `Co-Authored-By` trailer naming that address.
