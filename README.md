# alphaUniverse

The embedding-based virtual astronomical observatory.

alphaUniverse wraps [AION](https://huggingface.co/polymathic-ai/aion-base),
Polymathic's foundation model for astronomy, in a web interface for exploring
what it has learned: a 2-d projection of every galaxy in the field, and
patch-level similarity search across the survey.

## What you can do with it

**Explore the embedding space.** Every galaxy appears as a point in a parametric
UMAP projection, coloured by Galaxy Zoo morphology. Two projections share the
same space: *mean*, one point per galaxy from its average embedding, and *full*,
one point per embedding — so images and spectra from four surveys land in one
map. Filtering by morphology runs in the browser.

**Inspect a galaxy.** Selecting a point opens its Legacy Survey cutout, its
morphology label, which surveys it was crossmatched into, and the codebook token
behind each of its 576 image patches.

**Search by patch.** Click patches on the grid to build a query. The query is the
mean of the chosen patch embeddings; every galaxy scores the cosine similarity of
each of its own patches against it, and is ranked by its best-scoring patch.
Results come back as a score map per galaxy, which can be thresholded into a mask
and inverted.

## Data

`hanbangwu/alphauniverse-cosmos` is Legacy Survey DR10 south over the COSMOS
field, crossmatched against four more catalogues. Legacy Survey is the anchor:
every galaxy has one, the rest are present where the crossmatch found something.

| Survey             | Tokens per galaxy |
| ------------------ | ----------------- |
| Legacy Survey DR10 | 576 + 12 scalars  |
| HSC PDR3           | 576 + 13 scalars  |
| DESI EDR SV3       | 273               |
| SDSS               | 273               |

Every token carries a 768-d embedding, in two flavours: **encoded**, the
encoder's contextualised output, and **codebook**, the raw vector behind the
token. The 576 image tokens are a 24×24 patch grid over a 96-pixel centre crop.

## Running

Requires [uv](https://docs.astral.sh/uv/) and [bun](https://bun.sh).

### Frontend

```sh
cd frontend
cp .env.example .env     # point PUBLIC_API_URL at an API
bun install
bun run dev
```

### Backend

The API serves artifacts built by three Modal jobs, which must run in order:

```sh
uv run modal run modal_app.py::generate_embeddings   # encode every galaxy (GPU, ~3h)
uv run modal run modal_app.py::generate_index        # build the search index
uv run modal run modal_app.py::generate_projections  # fit and apply the projection
uv run modal deploy modal_app.py                     # serve
```

Pushes to `main` deploy automatically once CI passes.

### Working on it locally

Tests and benchmarks run against a synthetic artifact tree, with no GPU, no Modal
and no network:

```sh
uv run pytest
uv run python -m scripts.benchmark --galaxies 32
```

After changing a route or model in `app/main.py`, regenerate the API schema the
frontend's client is built from:

```sh
uv run python -m scripts.openapi     # updates frontend/openapi.json
cd frontend && bun run check         # regenerates src/lib/api and typechecks
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the pieces fit together
- [`docs/pipeline.md`](docs/pipeline.md) — the build stages and artifact schemas
- [`docs/performance.md`](docs/performance.md) — cost model, measurements, ceilings
- [`docs/testing.md`](docs/testing.md) — tests, benchmarks and CI
