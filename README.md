# alphaUniverse

The embedding-based virtual astronomical observatory.

## Data

`hanbangwu/alphauniverse-cosmos` is COSMOS crossmatched with Legacy Survey DR10 south as the anchor.

| Survey             | Tokens per galaxy |
| ------------------ | ----------------- |
| Legacy Survey DR10 | 576 + 12 scalars  |
| HSC PDR3           | 576 + 13 scalars  |
| DESI EDR SV3       | 273               |
| SDSS               | 273               |

Every token carries a 768-d embedding vector, in two flavours:

- Encoded: the encoder's output (contextualized)
- Codebook: the raw embedding vector behind each token from the codebook

## Projection

A Parametric UMAP runs the mean (the average of all embeddings per galaxy; one point per galaxy) and full (one point per embedding per galaxy (every modality in the same space)). Galaxies are coloured by GZ10 morphology.

## Details Panel

Selecting a galaxy opens its detail panel with the Legacy Survey cutout, its morphology label, and which surveys it was crossmatched with, and two views:

- Image: the cutout on its own
- Tokens: the codebook token behind each patch

## Patch Similarity

Click patches on the grid to build a query. The query embedding is the mean of all the query patches. Every galaxy scores the cosine similarity of each of its patches against the query, and is ranked by its highest-scoring patch. Masking thresholds the scores and is invertible.

## API

- `GET /meta` — dataset, revision, galaxy count, grid size, points files, class counts
- `GET /artifacts/{role}` — build artifacts
- `GET /galaxies/{galaxy}/image.png` — centre crop
- `GET /galaxies/{galaxy}/tokens` — codebook token per patch
- `GET /galaxies/{galaxy}/coverage` — one flag per survey
- `GET /similarity` — patch search

## Running it

```sh
uv run python -m scripts.openapi                    # updates frontend/openapi.json
cd frontend && bun run check                        # regenerates src/lib/api and type checks
uv run modal run modal_app.py::generate_embeddings  # build embeddings
uv run modal run modal_app.py::generate_index       # build the search index
uv run modal run modal_app.py::generate_projections # build projections
uv run modal deploy modal_app.py                    # serve
```

## Wishlist

- Fine-Tuning-as-a-Service
- Colour by something other than morphology (perhaps redshift)
- Hover a token in the embedding space and see where it sits in the projection view
- Cover a larger area of sky
