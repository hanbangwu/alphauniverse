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
- `predictions` holds random distributions in the stored forms; `prediction_basis` has a uniform mean and random orthonormal directions orthogonal to the all-ones vector, with spectrum token coefficients a few 1e-4 across, so every reconstructed spectrum token stays a distribution. They test the arithmetic, not production's spread of coefficients.
- `codebook` and `parametric_umap` are left out: nothing served reads them, and their absence exercises the 404 path.

`tests/test_pql.py` checks `app.pql`'s mode sums and maps against dense distributions rebuilt from the fixture's `predictions`.

`test_selected_table_values_join_the_direction` checks the query direction and `table_value_maps` against the stored embeddings of a selected image token, a Legacy Survey table value and an HSC table value.

The pipeline modules' tests need the `build` group, which every command above installs:

```sh
uv run --group build pytest tests/test_encode.py tests/test_predictions.py tests/test_compression.py tests/test_parametric_umap.py tests/test_alignment.py
```

- They build AION and its codecs with random weights from the configs in `tests/aion/`, copied from `polymathic-ai/aion-base`. No test downloads weights. The redshift codec built this way bins 0 to 1, its code default; the released weights replace that with 0 to 6.
- `tests/test_predictions.py` runs `generate_predictions` in its own tree on a copy of the shared tree's `tokens`: rewriting the shared `predictions` would break the memory map `app.pql` holds.
- `tests/test_compression.py` checks every scheme in `scripts/benchmarks/compression.py` on synthetic distributions: its fast overlap equals the overlap of what it decodes, lossless schemes return the distribution, quantised values are within half a step, and the current scheme takes the bytes `predictions` gives each mode.
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
