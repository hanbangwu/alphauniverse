# alphaUniverse

The embedding-based virtual astronomical observatory.

alphaUniverse wraps [AION](https://arxiv.org/abs/2510.17960), Polymathic's foundation model for astronomy, in a web interface for exploring its embeddings.

## What you can do with it

- **Explore the embedding space** - Every galaxy appears as a point in a parametric UMAP projection. There are two views: **mean**, one point per galaxy from its average embedding; and **full**, one point per embedding (all modalities share one map).
- **Inspect a galaxy** - Selecting a point opens its Legacy Survey cutout, its morphology label, which surveys it was crossmatched into, and the codebook token behind each of its 576 image patches.
- **Search by patch** - Click patches on the grid to find similar patches and galaxies by cosine similarity. Thresholding also gives zero-shot segmentation.

## Data

`hanbangwu/alphauniverse-cosmos` is Legacy Survey DR10 south over COSMOS crossmatched with Legacy Survey as anchor against four othe rcatalogues:

| Survey             | Modality | Tokens per galaxy |
| ------------------ | -------- | ----------------- |
| Legacy Survey DR10 | image    | 576 + 12 scalars  |
| HSC PDR3           | image    | 576 + 13 scalars  |
| DESI EDR SV3       | spectrum | 273               |
| SDSS               | spectrum | 273               |

Every token carries a 768-d embedding, in two flavours: **encoded**, the encoder's contextualised output, and **codebook**, the raw vector behind the token.

## Running

Requires [uv](https://docs.astral.sh/uv/) and [bun](https://bun.sh).

### Frontend

```sh
cd frontend
bun install
bun run dev
```

### Backend

The API serves artifacts built by three Modal jobs, which must run in order:

```sh
uv run modal run modal_app.py::generate_embeddings   # encode dataset
uv run modal run modal_app.py::generate_index        # build the search index
uv run modal run modal_app.py::generate_projections  # fit and apply the projection
uv run modal deploy modal_app.py                     # serve
```

Pushes to `main` deploy automatically once CI passes.

### Working on it locally

```sh
uv run pytest
uv run python -m scripts.benchmark --galaxies 32
```

After changing a route or model in `app/main.py`, regenerate the API schema the frontend's client is built from:

```sh
uv run python -m scripts.openapi     # updates frontend/openapi.json
cd frontend && bun run check         # regenerates src/lib/api and typechecks
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md): how the pieces fit together
- [`docs/pipeline.md`](docs/pipeline.md): the build stages and artifact schemas
- [`docs/performance.md`](docs/performance.md): cost model, measurements, ceilings
- [`docs/testing.md`](docs/testing.md): tests, benchmarks and CI
