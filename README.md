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

## Running

```sh
uv run python -m scripts.openapi                    # updates frontend/openapi.json
cd frontend && bun run check                        # regenerates src/lib/api and type checks
uv run modal run modal_app.py::generate_embeddings  # generate embeddings
uv run modal run modal_app.py::generate_index       # generate search index
uv run modal run modal_app.py::generate_projections # generate projections
uv run modal deploy modal_app.py                    # serve
```
