# alphaUniverse

The embedding-based virtual astronomical observatory.

## The data

`hanbangwu/alphauniverse-cosmos` is the COSMOS field crossmatched with Legacy Survey DR10 south as the anchor.

Each galaxy is encoded in a single pass over every modality it has:

| Survey                       | Tokens per galaxy |
| ---------------------------- | ----------------- |
| Legacy Survey DR10 (g/r/i/z) | 576 + 12 scalars  |
| HSC PDR3 (g/r/i/z/y)         | 576 + 13 scalars  |
| DESI EDR SV3                 | 273               |
| SDSS                         | 273               |

- Images are 96 px centre crops on a 24x24 grid
- Every galaxy is guaranteed to have at least Legacy Survey as it's the anchor.
- Either a whole survey is matched or none of one survey is matched.
- Every token carries a 768-d embedding vector, in two flavours:
  - **Encoded** — the encoder's output (contextualized)
  - **Codebook** — the raw embedding vector behind each token from the codebook

## The atlas

- Laid out by a **parametric UMAP**
- Two point sets: **mean** (the average of all embeddings per galaxy; one point per galaxy) and **full** (one point per embedding per galaxy (every modality in the same space))
- Coloured by GZ10 morphology, with a separate swatch for unlabelled galaxies
- Filter by morphology class, with per-class counts
- Click a point to select its galaxy

## Galaxy details

Selecting a galaxy opens its panel: the Legacy Survey cutout, its morphology label, and which surveys it was crossmatched with. Two views:

- **Image** — the cutout on its own
- **Tokens** — the codebook token behind each patch, coloured by how often it recurs in this galaxy

## Patch similarity

Opens from the details panel. Click patches on the grid to build a query, then:

- **Search space** — encoded embeddings, codebook embeddings, or a weighted blend of both
- **Multiple patches**
  - "and" takes the mean of the query patches and performs search, and
  - "or" searches for each of the query patches then takes the mean of the results
- **Masking** — threshold the scores, invertible (applied in the frontend live)
- **Top-k / show top** — how many of a galaxy's best patches are averaged into its score, and how many galaxies come back

Every search covers the whole dataset. It ranks each galaxy by the mean of its best-scoring patches and returns the top matches with their images and similarity maps. The query galaxy rides along at the head of the result — it is never ranked against the matches, it is there so one colour scale spans it and every match and the maps stay comparable.

## Downloads and cache

**Download embeddings** in the left panel gives `encoded.parquet` for the current build and dataset. One row per galaxy:

| Column                      | Arrow type                              | DuckDB type    |
| --------------------------- | --------------------------------------- | -------------- |
| `galaxy`                    | `int32`                                 | `INTEGER`      |
| `ls`, `hsc`, `desi`, `sdss` | `list<fixed_size_list<halffloat, 768>>` | `FLOAT[768][]` |
| `gz10`, `provabgs`          | `bool`                                  | `BOOLEAN`      |

`galaxy` and the two flags are always written; an unmatched survey's cell is `NULL` and takes no space on disk.

Cells hold tokens grouped by survey, split by the modality each encoder token came from — for `ls`, 576 image patches then up to 12 scalars.

`codebook.parquet` and `tokens.parquet` has similar shape.

## API

- `GET /meta` — dataset, revision, galaxy count, grid size, points files, class counts
- `GET /artifacts/{role}` — a build artifact, as written
- `GET /galaxies/{galaxy}/image.png` — centre crop
- `GET /galaxies/{galaxy}/tokens` — codebook token per patch
- `GET /galaxies/{galaxy}/coverage` — one flag per survey
- `GET /similarity` — patch search, as an Arrow IPC stream: one row per galaxy, each carrying both maps

## Running it

```sh
uv run modal run modal_app.py::generate_embeddings  # build embeddings
uv run modal run modal_app.py::generate_projections # build projections
uv run modal deploy modal_app.py                    # serve
```

For OpenAPI:

```sh
uv run python -m scripts.openapi # rewrites frontend/openapi.json
cd frontend && bun run check     # regenerates src/lib/api and typechecks
```

## Wishlist

- Fine-Tuning-as-a-Service
- Colour by something other than morphology (perhaps redshift)
- Hover a token in the embedding space and see where it sits in the projection view
- Cover a larger area of sky
