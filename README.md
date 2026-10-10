# alphaUniverse

The embedding-based virtual astronomical observatory.

alphaUniverse wraps [AION](https://arxiv.org/abs/2510.17960), Polymathic's foundation model for astronomy, in a web interface for exploring its embeddings.

## What you can do with it

- **Explore the embedding space** - Every galaxy appears as a point in a parametric UMAP projection. There are two views: **mean**, one point per galaxy from its average embedding; and **full**, one point per embedding (all modalities share one map). Hovering or selecting a point shows the galaxy's morphology and image.
- **Inspect a galaxy** - Selecting a point shows its Legacy Survey image or its DESI spectrum, smoothed for display, its morphology label, and which surveys it was crossmatched into.
- **Search by token** - Search on a selected galaxy shows the token behind each of its 576 image patches and each span of its spectrum. A Tabular Data table lists the galaxy's catalogue values; the 26 that AION encodes (25 photometric values and one redshift) can be checked, and after a search are shaded by their score. Click patches, spans, values, or any mix, then press Search to find galaxies whose tokens are closest by cosine similarity. Each match shows a heatmap of its patch scores, and of its span scores where it has a spectrum; thresholding the patch heatmap gives zero-shot segmentation.
- **Search by text** - The Text Search tab beside the projection takes a description and lists the galaxies whose AION embeddings, mapped into [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-2)'s space, lie nearest it, each with its cosine score and morphology. Clicking one selects it.

## Data

`hanbangwu/alphauniverse-cosmos` is Legacy Survey DR10 south over COSMOS, crossmatched against HSC, DESI, SDSS, Galaxy Zoo 10 and PROVABGS. The image and spectrum surveys are tokenised:

| Survey             | Modality | Tokens per galaxy |
| ------------------ | -------- | ----------------- |
| Legacy Survey DR10 | image    | 576 + 12 scalars  |
| HSC PDR3           | image    | 576 + 13 scalars  |
| DESI EDR SV3       | spectrum | 273               |
| SDSS               | spectrum | 273               |

A galaxy with a spectroscopic redshift also gets one redshift token: DESI's if it is unflagged and at most 6, else SDSS's under the same test, else none.

Every token carries a 768-d embedding, in two flavours: **encoded**, the contextualised encoding AION's decoder reads, and **codebook**, the encoder's input embedding of the token id, before any context.

## Running

Requires [uv](https://docs.astral.sh/uv/) and [bun](https://bun.sh).

### Local API

Serves the API from a small synthetic artifact tree in the production schemas:

```sh
uv run python -m scripts.fixture                                   # writes .cache/fixture
ALPHAUNIVERSE_CACHE=.cache/fixture uv run fastapi dev app/main.py  # serves 127.0.0.1:8000
```

The fixture's 12 galaxies and their embeddings are synthetic: develop against it, never measure with it. Images, spectra and the catalogue table read the Hugging Face dataset itself: the first such request downloads it into the local Hugging Face cache (24.33 GB at the pinned revision), and with `HF_HUB_OFFLINE=1` they fail unless it is already there. The first text search downloads EmbeddingGemma the same way.

### Frontend

```sh
cd frontend
bun install
bun run dev
```

The dev build calls the API at `http://127.0.0.1:8000`, so start the local API first.

### Pipeline

```sh
uv run modal run modal_app.py::generate_embeddings        # encode dataset
uv run modal run modal_app.py::generate_index             # build the search index
uv run modal run modal_app.py::generate_projections       # fit and apply the projection
uv run modal run modal_app.py::generate_pairs             # AION and EmbeddingGemma embeddings of each galaxy
uv run modal run modal_app.py::generate_alignment         # fit the AION to EmbeddingGemma map
uv run modal run modal_app.py::generate_aion_gemma_space  # every galaxy's vector for text search
```

### API schema

After changing a route or model in `app/main.py`, regenerate the API schema the frontend's client is built from:

```sh
uv run python -m scripts.openapi     # updates frontend/openapi.json
cd frontend && bun run check         # regenerates src/lib/api and typechecks
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md): how the pieces fit together
- [`docs/pipeline.md`](docs/pipeline.md): the build stages and artifact schemas
- [`docs/benchmarks.md`](docs/benchmarks.md): benchmarks, measurements, ceilings
- [`docs/testing.md`](docs/testing.md): tests and CI
