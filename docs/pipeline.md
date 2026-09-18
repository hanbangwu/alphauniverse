# Build pipeline

Three Modal jobs, run in order. Each consumes the previous one's output from a
shared volume mounted at `/cache`, and `ALPHAUNIVERSE_CACHE` points the app at
it. Artifacts land under `$ALPHAUNIVERSE_CACHE/<author>/<name>/<revision>/`, so
changing `DATASET_REVISION` switches trees rather than overwriting one.

```sh
uv run modal run modal_app.py::generate_embeddings   # L4, ~3h
uv run modal run modal_app.py::generate_index        # CPU
uv run modal run modal_app.py::generate_projections  # L4
uv run modal deploy modal_app.py
```

Stages 1 and 3 need the `build` dependency group (torch, AION, umap-learn,
wandb); the serving image does not install it.

## The dataset

`hanbangwu/alphauniverse-cosmos` is built by `scripts/alphauniverse_cosmos.py`:
Legacy Survey DR10 south, cut to a √2° box on the COSMOS field, left-joined
against five catalogues with LSDB and pushed to the Hub. Legacy Survey is the
**anchor** — every row has one, and the other surveys are present only where the
crossmatch found something.

| Survey             | Modality | Tokens per galaxy |
| ------------------ | -------- | ----------------- |
| Legacy Survey DR10 | image    | 576 + 12 scalars  |
| HSC PDR3           | image    | 576 + 13 scalars  |
| DESI EDR SV3       | spectrum | 273               |
| SDSS               | spectrum | 273               |

576 image tokens is a 24×24 patch grid (`GRID`, `N_PATCHES`) over a 96-pixel
centre crop. That grid is the unit of interaction in the UI.

The script is not part of the deployed pipeline and needs `lsdb`, which is not a
project dependency — run it with `uv run --with datasets --with lsdb`.

## Stage 1 — `generate_embeddings`

For each galaxy: tokenise every modality it has, run all its tokens through the
AION encoder in one pass, then split the output back apart by modality id.

Three stores fall out, all with the same schema:

```
galaxy: int32
ls, hsc, desi, sdss: list<fixed_size_list<float16, 768>>   -- null where unmatched
gz10, provabgs:      bool
```

- **`encoded`** — the encoder's contextualised output. Because every modality is
  encoded together, a galaxy's spectrum tokens carry information from its image.
- **`codebook`** — the raw codebook vector behind each token, uncontextualised.
- **`tokens`** — the token ids, `list<uint32>` instead of embeddings.

Within a cell, image or spectrum tokens come first and the survey's scalars
follow. Everything that slices `[:576]` depends on that order.

Rows are written 1024 at a time into `.partial` files that replace the real
artifacts only at the end, so an interrupted run leaves the previous artifacts
intact.

## Stage 2 — `generate_index`

Builds `IVF{nlist},SQfp16` over the anchor survey's **image patches only** —
scalars are sliced off — with inner product as the metric and rows L2-normalised
first, so inner product is cosine similarity.

Vectors are added galaxy by galaxy in row order, which is what makes
`id = galaxy * 576 + patch` true. Nothing may add to or reorder the index
without breaking `app/search.py`.

`nlist` defaults to `NLIST` (16384), capped so training keeps at least
`MIN_TRAIN_PER_CENTROID` vectors per centroid. At production scale the cap does
nothing; it is what lets a 12-galaxy fixture build. Training uses at most
`TRAIN_GALAXIES` (2048) galaxies, capped at the dataset size.

## Stage 3 — `generate_projections`

Fits a parametric UMAP — a 768→256→128→2 MLP trained to reproduce UMAP's
fuzzy-simplicial-set structure — on a `SAMPLE` of 500k embeddings, then applies
it to everything. Being a function rather than a fitted table is the point: the
same model maps both point sets, so they share one space.

Two passes over `encoded`:

1. Accumulate each galaxy's mean embedding and draw the training sample.
2. Project every embedding individually.

Two outputs, both in the `POINTS` schema (`app/config.py`):

```
galaxy: int32 not null
x, y:   float32 not null
category: uint8   -- GZ10 morphology, null where unlabelled
```

- **`mean_points`** — one row per galaxy, from its mean embedding. Small, loaded
  on first paint, and the source of `/meta`'s label counts.
- **`full_points`** — one row per embedding, every modality in the same space.
  Roughly 600× larger; loaded only when the user asks for it.

Training logs to Weights & Biases when `WANDB_API_KEY` is set and is otherwise
disabled.

## Fixtures

`scripts/fixture.py` writes the same schemas at a size that fits in CI, which is
what the tests and benchmarks run against. Embeddings are drawn from 64 random
cluster centres with noise, not uniform noise — in 768 dimensions uniform
vectors are all near-orthogonal, which would make every ranking arbitrary and
every recall number meaningless.

It skips `codebook` and `parametric_umap` (nothing served reads them, and their
absence exercises the 404 path) and substitutes a fixed random 2-d projection for
the UMAP, so the test path never needs torch.

```sh
uv run python -m scripts.fixture --galaxies 12 --out .cache/fixture
```
