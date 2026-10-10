# Testing

Benchmarks are in `docs/benchmarks.md`.

## Tests

```sh
uv run --group build pytest
uv run --group build pytest -k search  # one area
uv run ruff check app scripts tests modal_app.py
uv run ruff format app scripts tests modal_app.py
```

Tests need no Modal, GPU or network. `tests/test_dataset.py` replaces `dataset()` with a two-row `Dataset`; the image and spectrum endpoints are untested. `tests/conftest.py` builds one synthetic tree per session with `scripts/fixture.py`, in the production schemas:

- Embeddings cluster around fixed random centres. Uniform noise in 768 dimensions is nearly orthogonal, which would make every ranking arbitrary.
- The 2-d points come from one fixed random projection in place of the trained parametric UMAP. One projection serves both point sets, as the projector does in production.
- `aion_gemma_space` holds random unit vectors. The text search tests replace `embed_queries`, so no test loads EmbeddingGemma.
- `codebook` and `parametric_umap` are left out: nothing served reads them, and their absence exercises the 404 path.

Five tests in `tests/test_search.py` check what the shared tree cannot show:

- `test_approximate_ranking_agrees_with_exact` compares `search()` with `exact_ranking` in `scripts/benchmarks/search_quality.py`: the same search without the candidate step, a brute force over every token's float32 embedding in place of the index's fp16 copies. It asks for fewer matches than the corpus holds, so the candidate step must choose. It checks the galaxies, their order, each score and each image token, spectrum token and scalar map (null where the galaxy has no spectrum or HSC match) to within `SCORE_TOLERANCE`, after asserting that the exact scores are more than twice that apart. On production data the two need not agree; `search_quality` measures how often they do.
- `test_ids_stay_contiguous_across_add_batches` builds its own index with one galaxy per `add()` call, galaxies without a spectrum included, since the shared tree goes in with one call. A reordered or dropped batch would silently shift every galaxy id.
- `test_embeddings_that_are_not_finite_are_rejected` passes `image_tokens()` a cell of infinities, which the shared tree never holds. `faiss.normalize_L2` would turn them into NaN silently.
- `test_rank_keeps_the_query_first_and_each_row_together` calls `rank()` on arrays built in the test, since on the shared tree `candidates()` already returns galaxies in ranked order.
- `test_a_search_that_finds_too_few_looks_further` changes `PROBE` and `NPROBE` so that the first search falls short, since at their defaults it finds every galaxy on the shared tree. It asks for one vector over one list, 16,384 (`NLIST`) vectors over one list, more than the shared tree holds, and one vector over every list, and checks that the answer starts with the query galaxy and holds `matches` other galaxies, or all of them when the tree holds fewer, none twice.

`test_selected_scalars_join_the_direction` checks the query direction and `scalar_maps` against the stored embeddings of a selected image token, a Legacy Survey scalar and an HSC scalar.

The pipeline modules' tests need the `build` group, which every command above installs:

```sh
uv run --group build pytest tests/test_encode.py tests/test_parametric_umap.py tests/test_alignment.py
```

- They build AION and its codecs with random weights from the configs in `tests/aion/`, copied from `polymathic-ai/aion-base`. No test downloads weights. The redshift codec built this way bins 0 to 1, its code default; the released weights replace that with 0 to 6.
- `tests/test_parametric_umap.py` trains the projector for one epoch on a four-galaxy tree, twice, and checks the results match. It also checks that `scan` returns each chosen embedding in the order asked and rejects a sample past the streamed embeddings.
- `tests/test_alignment.py` checks that `fit_linear` recovers a known affine map and that `recall` is 1 for an exact prediction. Nothing loads EmbeddingGemma, so `generate_pairs`, `text_model` and `embed_queries` in `app/text_search.py` are untested.

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

1. **python**: `ruff check`, `ruff format --check`, then `uv run --group build pytest`
2. **openapi**: regenerates `frontend/openapi.json` and fails if it differs from the committed copy
3. **frontend**: `bun run lint` and `bun run check`
4. **deploy**: Modal, on pushes to `main` once the other three pass

Every uv command runs with `UV_LOCKED=1`, so a `pyproject.toml` change without its `uv.lock` fails.

Lint fails on a comment starting with `TODO`, `FIXME`, `HACK` or `XXX`, in any case: ruff's `FIX` rules check Python, and eslint's `no-warning-comments` checks JavaScript, TypeScript and Svelte `<script>`, after any leading `*`. Other files, and Svelte markup and styles, are not checked.

`benchmark.yml` runs only when started by hand (`docs/benchmarks.md`).
