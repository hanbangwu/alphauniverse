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

- `test_approximate_ranking_agrees_with_exact` compares `search()` with a brute-force ranking over every token. The brute force follows `search()` except for the candidate step, scores the float32 embeddings rather than the index's fp16 copies, and holds the whole corpus in memory, so it only runs at fixture scale. The test asks for fewer matches than the corpus holds, so the candidate step has to choose, and checks the galaxies, their order, and each score to within `SCORE_TOLERANCE`. It first asserts that the exact scores it compares are more than twice that tolerance apart, so rounding within the tolerance cannot reorder them. It also checks each galaxy's patch and span score maps to within the same tolerance, with a null span map where the galaxy has no spectrum: a map entry is a token's score, and the galaxy's score is the largest of them. The fixture's clusters are tight enough that the index finds the tokens the brute force ranks highest, so the two agree; on production data they would not.
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

`.github/workflows/ci-cd.yml` runs on pull requests and on pushes to `main`, except a push that changes only `CLAUDE.md`, `.claude/rules/`, `README.md` or `docs/`, whose pull request already ran the checks:

1. **python**: `ruff check`, `ruff format --check`, then `uv run pytest`
2. **openapi**: regenerates `frontend/openapi.json` and fails if it differs from the committed copy
3. **frontend**: `bun run lint` and `bun run check`
4. **deploy**: Modal, on pushes to `main` only, and only if the other three pass

Every uv command in the workflow runs with `UV_LOCKED=1`, so a `pyproject.toml` change without a matching `uv.lock` fails CI instead of being re-resolved.

Lint fails on a comment that starts with `TODO`, `FIXME`, `HACK` or `XXX`, in any case: ruff's `FIX` rules check Python comments, and eslint's `no-warning-comments` checks JavaScript, TypeScript and Svelte `<script>` comments, after any leading `*`. Comments in other files and in Svelte markup or styles are not checked.

`.github/workflows/build.yml` installs the `build` group and runs the torch tests on pull requests that change the torch modules or what they import (`app/encode.py`, `app/parametric_umap.py`, `app/config.py`, `app/dataset.py`, `app/search.py`), what the tests build their trees with (`app/cutouts.py`, `app/spectra.py`, `scripts/fixture.py`), `pyproject.toml`, `uv.lock`, those tests, `tests/aion/` or the workflow itself.

`.github/workflows/rules.yml` runs on pull requests. Its Rules job fails on a commit in the pull request authored by `noreply@anthropic.com`, a GitHub `[bot]` account or Copilot, and on one committed by `noreply@anthropic.com` without a `Co-Authored-By` trailer naming that address.
