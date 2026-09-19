# Build pipeline

Four Modal jobs; the README has the commands. Embeddings, index and projections run in that order, each consuming the previous one's output; cutouts read the source dataset directly and can run at any point. All of them share a volume mounted at `/cache`, and `ALPHAUNIVERSE_CACHE` points the app at it. Artifacts are written under `$ALPHAUNIVERSE_CACHE/<author>/<name>/<revision>/`, so changing `DATASET_REVISION` switches trees rather than overwriting one.

`generate_embeddings` and `generate_projections` need the `build` dependency group (torch, AION, umap-learn, wandb); the serving image does not install it.

## The dataset

`hanbangwu/alphauniverse-cosmos` is built by `scripts/alphauniverse_cosmos.py`: Legacy Survey DR10 south, cut to a √2° box on the COSMOS field, left-joined against five catalogues with LSDB and pushed to the Hub. The README lists the surveys and their token counts.

The script is not part of the deployed pipeline and needs `lsdb`, which is not a project dependency; run it with `uv run --with datasets --with lsdb`.

## `generate_embeddings`

For each galaxy: tokenise every modality it has, run all its tokens through the AION encoder in one pass, then split the output back apart by modality id.

Three stores are written, all with the same schema:

```
galaxy: int32
ls, hsc, desi, sdss: list<fixed_size_list<float16, 768>>   -- null where unmatched
gz10, provabgs:      bool
```

- **`encoded`**: the encoder's contextualised output. Because every modality is encoded together, a galaxy's spectrum tokens carry information from its image.
- **`codebook`**: the raw codebook vector behind each token, uncontextualised.
- **`tokens`**: the token ids, `list<uint32>` instead of embeddings.

Within a cell, image or spectrum tokens come first and the survey's scalars follow.

## `generate_index`

Builds `IVF{nlist},SQfp16` over the anchor survey's **image patches only** (the scalars are sliced off), with inner product as the metric and rows L2-normalised first, so inner product is cosine similarity. `app/search.py` states the id layout and how `nlist` is chosen.

## `generate_cutouts`

Centre-crops every galaxy's anchor image to `CROP_PX` square and PNG-encodes it, in dataset row order:

```
galaxy: int32
png:    large_binary
```

Row `g` is galaxy `g`; `app/cutouts.py` states why that matters and checks it on load. The serving app will not start without this artifact.

## `generate_projections`

Fits a parametric UMAP on a sample of embeddings and applies it to every one; `app/parametric_umap.py` describes the model and the two passes over `encoded`.

Two outputs, both in the `POINTS` schema (`app/config.py`):

```
galaxy: int32 not null
x, y:   float32 not null
category: uint8   -- GZ10 morphology, null where unlabelled
```

- **`mean_points`**: one row per galaxy, from its mean embedding. Small, loaded on first paint, and the source of `/meta`'s label counts.
- **`full_points`**: one row per embedding, every modality in the same space. Roughly 600× larger; loaded only when the user asks for it. Written survey by survey, so its galaxy column is not monotonic.

Training logs to Weights & Biases when `WANDB_API_KEY` is set and is otherwise disabled.
