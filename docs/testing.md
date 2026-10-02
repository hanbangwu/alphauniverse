# Testing and benchmarking

Tests run against a synthetic artifact tree built by `scripts/fixture.py`; they need neither Modal, a GPU nor the network. The torch tests, described below, also build small trees of their own.

The tree has the production schemas at a size that fits in a CI runner, so the API and the search path run real code against real files:

- Embeddings are drawn around a fixed set of random cluster centres, not as uniform noise. In 768 dimensions uniform random vectors are all nearly orthogonal, which would make every ranking arbitrary and every recall figure meaningless.
- The 2-d points come from a fixed random projection, standing in for the trained parametric UMAP, so the default suite runs without torch. One projection serves both point sets, as one trained projector does in production.
- The tree leaves out `codebook` and `parametric_umap`: nothing served reads them, and their absence exercises the 404 path.

## Tests

```sh
uv run pytest
uv run pytest -k search  # one area
uv run ruff check app scripts tests modal_app.py
uv run ruff format app scripts tests modal_app.py
```

`tests/conftest.py` builds the tree once per session.

Three tests in `tests/test_search.py` check what the shared tree cannot show on its own:

- `test_approximate_ranking_agrees_with_exact` compares `search()` with a brute-force ranking over every token, `exact_ranking` in `scripts/recall.py`. The brute force follows `search()` except for the candidate step, scores the float32 embeddings rather than the index's fp16 copies, and holds the whole corpus in memory, so the suite runs it only at fixture scale; `scripts/recall.py` runs it on the production artifacts (Benchmarks, below). The test asks for fewer matches than the corpus holds, so the candidate step has to choose, and checks the galaxies, their order, and each score to within `SCORE_TOLERANCE`. It first asserts that the exact scores it compares are more than twice that tolerance apart, so rounding within the tolerance cannot reorder them. It also checks each galaxy's patch and span score maps to within the same tolerance, with a null span map where the galaxy has no spectrum: a map entry is a token's score, and the galaxy's score is the largest of them. The fixture's clusters are tight enough that the index finds the tokens the brute force ranks highest, so the two agree; on production data they need not, which is what `scripts/recall.py` measures.
- `test_ids_stay_contiguous_across_add_batches` builds its own index with one galaxy per `add()` call, including galaxies without a spectrum, because the shared tree is smaller than `BATCH` and goes in with a single call. A reordered or dropped batch would shift every galaxy id in every `/similarity` response without an error.
- `test_rank_keeps_the_query_first_and_each_row_together` calls `rank()` on arrays built in the test, because on the shared tree `candidates()` already returns every query's galaxies in ranked order, so `rank()` never reorders them. In them the query galaxy scores below two others, and a galaxy without a spectrum, whose patch scores are all negative, moves from the middle to the end.

The torch modules have their own tests, which skip unless the `build` group is installed:

```sh
uv run --group build pytest tests/test_encode.py tests/test_parametric_umap.py
```

They build AION and its codecs with random weights from the configs in `tests/aion/`, copied from `polymathic-ai/aion-base` at the revision in `tests/aion/REVISION`, so no test downloads weights. `test_copied_configs_come_from_the_pinned_revision` fails when that revision is not `AION_REVISION`; moving the pin means copying the configs again and updating that file. `test_saved_weights_load_back_unchanged` saves that small AION with `save_pretrained` and loads it back with `from_pretrained`. Both need `safetensors`, the format `aion-base` stores its weights in, which the `build` group gets through aion's `torch` extra. `tests/test_parametric_umap.py` trains the projector for one epoch on a four-galaxy tree, twice, to check that the result is deterministic. `test_padded_sdss_spectra_keep_their_flux` is marked `xfail(strict=True)`: SDSS spectra end in `lambda = -1` padding that zeroes the codec input, and once that is fixed the test passes and the mark has to go.

## Benchmarks

```sh
uv run modal run -m scripts.benchmark
uv run modal run -m scripts.benchmark --runs 50
```

The benchmark measures the production artifacts on the Modal volume, never the fixture. `modal run` starts an ephemeral copy of `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency. A client in a separate container times one cold `/meta` and one cold `/similarity`, then, warm, `/meta`, the image, tokens and coverage endpoints, and patch-only `/similarity` at three values of `matches`. The spectrum endpoints, span queries and artifact downloads are not timed. A container with the server's spec times the startup loads, then the stages of `search()` for the first query and warm, for two or three versions of the code on the same host: the checked-out commit (after), the head of `origin/main` (before), and the best version the stored report names (best), unless its code matches one of the other two. The benchmark refuses to run with uncommitted changes, and fetches `origin/main` first. Each version is sent as a `git archive` of `app/`, `scripts/` and `modal_app.py`, and runs its own `stages` in a subprocess, in mirrored order: after, before, best, then back again. The after version goes first, so its first round is the one with a cold page cache. All versions use the checked-out commit's locked dependencies. A round that fails, for example because its version cannot read the current artifacts, is reported with its error. Request latency is timed for the checked-out commit only.

The run writes its report to `docs/benchmarks/latest.json`, replacing the previous one; commit it before the next run. The report records the commits, dataset revision of each round, date, the server's CPU, memory and concurrency settings (`max_inputs`, `max_containers` and `scaledown_window`, from the constants in `modal_app.py` that `fastapi_app` is defined with), the client's Modal spec, the thread configuration, and the CPU identity (vendor, family, model and model name) of the client and stage containers. It names the best version: the lowest warm `search()` time at 32 matches (`total_p50_ms`), averaged over the version's rounds, among versions whose rounds all succeeded. A stored best commit that is not in the clone is skipped, noted and dropped. If the request timings or the stage container fail as a whole, for example on a timeout, the report records that part's error and keeps the other part, and the run still writes it. When no version's rounds all succeeded, the report keeps the stored best commit, without its figure from the earlier run, and notes it, so the next run still times it. It never runs in CI.

```sh
uv run modal run -m scripts.recall
uv run modal run -m scripts.recall --per-kind 50
```

`scripts/recall.py` measures how often `search()` finds the galaxies an exact search ranks highest, on the production artifacts. A container on the build image, with the build jobs' CPU and memory and the volume mounted read-only, loads every patch and span embedding once. It then runs `--per-kind` queries (100 by default) of each kind, each kind drawn with its own fixed seed: 4 patches (the benchmark's own queries), 4 spans, and 4 of each, at the default 32 matches. Spans are drawn from those that overlap the observed wavelength range of the galaxy's spectrum, DESI before SDSS as the index takes them, since the spectrum chart offers only those. A query's recall is the share of `exact_ranking`'s 32 galaxies that `search()` also returns, with `search()` at the served `PROBE` and `NPROBE`. The report gives each kind's mean, its minimum, the share of queries that found all 32, and the share where `search()` returned fewer than 32. Like the benchmark's, it records the commit, dataset revision, date, the container's Modal spec and the thread configuration, and adds the corpus load time, the total time and the peak memory.

Read `docs/performance.md` before drawing a conclusion from a benchmark or recall run.

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

`.github/workflows/build.yml` installs the `build` group and runs the torch tests on pull requests that change the torch modules or what they import (`app/encode.py`, `app/parametric_umap.py`, `app/config.py`, `app/dataset.py`, `app/search.py`), what the tests build their trees with (`app/cutouts.py`, `app/spectra.py`, `scripts/fixture.py`), `pyproject.toml`, `uv.lock`, those tests, `tests/aion/` or the workflow itself.

`.github/workflows/rules.yml` runs on pull requests. Its Rules job fails on a commit in the pull request authored by `noreply@anthropic.com`, a GitHub `[bot]` account or Copilot, and on one committed by `noreply@anthropic.com` without a `Co-Authored-By` trailer naming that address.
