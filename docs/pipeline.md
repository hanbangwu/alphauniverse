# Build pipeline

Six Modal jobs; the README has the commands. Embeddings come first, then the index and the projections, which read `encoded`. `generate_pairs` reads the dataset and `encoded`, `generate_alignment` reads `pairs`, and `generate_aion_gemma_space` reads `pairs` and `alignment`. All jobs share a volume mounted at `/cache`, and `ALPHAUNIVERSE_CACHE` points the app at it. Artifacts go under `$ALPHAUNIVERSE_CACHE/<author>/<name>/<revision>/`, so changing `DATASET_REVISION` switches trees rather than overwriting one.

Each job writes its artifacts in place, so a failed run leaves them incomplete; rerun the job. Every artifact holds one row per galaxy in dataset row order, and the app relies on that without checking it, so a build directory must come from one complete run of the pipeline.

Every job but `generate_index` needs the `build` dependency group (torch, AION, umap-learn, wandb); the serving image does not install it.

## The dataset

`hanbangwu/alphauniverse-cosmos` is built by `scripts/alphauniverse_cosmos.py`: Legacy Survey DR10 south, cut to a √2° box on the COSMOS field, left-joined against five catalogues with LSDB and pushed to the Hub. The README lists the surveys and their token counts.

The script is outside the deployed pipeline and needs `lsdb`, which is not a project dependency: run it with `uv run --with datasets --with lsdb`.

The serving app reads images and spectra from the dataset itself, from the copy `generate_embeddings` cached on the volume, with `HF_HUB_OFFLINE=1`. Offline, `datasets` ignores `DATASET_REVISION` and loads the most recently cached revision, so the volume must hold only the revision the artifacts were built from.

- **Image**: the anchor survey's `rgb` cutout, centre-cropped to `CROP_PIXELS` square and PNG-encoded on each request.
- **Spectrum**: wavelength in Ångström. Samples the survey pads with (wavelength at or below zero) are dropped, and samples it masks have NaN flux.

## `generate_embeddings`

For each galaxy: tokenise every modality it has, run all its tokens through the AION encoder in one pass, then split the output back apart by modality id.

The model and every codec load from the latest commit of `polymathic-ai/aion-base`, so a push to that repository changes the next build.

Galaxies are encoded one at a time and written in batches of 1024 rows.

Three stores are written, with the same columns:

```
galaxy: int32
ls, hsc, desi, sdss: list<fixed_size_list<float16, 768>>   -- null where unmatched
gz10, provabgs:      bool
```

- **`encoded`**: what AION's `_encode` returns, the context its decoder reads. That is the encoder's output after `encoder_norm`, mapped by the linear `decoder_proj_context`, with each token's position and modality embeddings added back. Those added embeddings depend only on the token's modality and position, so every galaxy shares them. Because every modality is encoded together, a galaxy's spectrum tokens carry information from its image.
- **`codebook`**: the encoder's input embedding of each token, before position and modality embeddings are added or any context is mixed in, so it depends only on the token id and its modality.
- **`tokens`**: the token ids, so each survey cell is a `list<uint32>` instead of a list of embeddings.

Within an image cell the patches come first and the survey's scalars follow. A spectrum cell leads with the codec's normalisation token, then holds one token per 25.6 Å from 3500 Å. AION resamples every spectrum onto 8704 pixels of 0.8 Å from 3500 Å and downsamples by 32, so a spectrum cell holds 273 tokens whatever survey it came from: the normalisation token and 272 spans.

## `generate_index`

Builds `IVF{nlist},SQfp16` over one block per galaxy, in galaxy order: the anchor survey's 576 **image patches**; then, if the galaxy has a spectrum, the 272 spectral tokens of its first matched spectrum survey, DESI before SDSS, with the normalisation token dropped; then the anchor survey's 12 **scalars**; then, if the galaxy has an HSC match, HSC's 13 scalars. HSC's image patches are not indexed. Inner product is the metric and rows are L2-normalised first, so inner product is cosine similarity. A row that is not finite fails the build.

A vector's id is its position in that sequence: galaxy `g` starts at `588 g + 272 s + 13 h`, where `s` and `h` count the galaxies before it that have a spectrum and an HSC match. The index does not store this layout: the app rebuilds it at startup from which galaxies have a spectrum and an HSC match in `tokens`, so it holds only while `tokens` and `encoded` agree on that. One `generate_embeddings` run writes both.

`generate_index` writes the direct map into the file, each vector's list and offset in 8 B, so `index()` reads the map in one sequential read instead of from every list's ids.

## `generate_projections`

Fits a parametric UMAP on a sample of embeddings and applies it to every one, in two passes over `encoded`, survey by survey. The first pass accumulates each galaxy's mean embedding over all its tokens, and draws `SAMPLE` (5,000,000) embeddings uniformly from the whole store. `train_test_split` holds out `VALIDATION` (30%) of the sample. The projector, an MLP from a normalised 768-d embedding to 2-d, trains on the rest: UMAP's fuzzy simplicial set over the training rows weights the edges between neighbours, and each step draws edges by weight, pulls their endpoints together, and pushes each edge's first endpoint away from `NEGATIVES` (5) random rows. Each epoch logs `train/loss` and `validation/loss` to Weights & Biases when `WANDB_API_KEY` is set, and logs nothing otherwise; on Modal the job reads it from the `wandb-secret` secret; the validation loss is the same loss over the held-out rows' own fuzzy simplicial set, without gradients. The second pass projects every embedding.

The trained projector, **`parametric_umap`**, is a `torch.save` of:

```
dim:   int         -- input width, DIM
state: state_dict  -- ParametricUMAP weights
```

Nothing reads it at serve time. The job then projects with the trained model and writes two point sets, both in the `POINTS` schema (`app/config.py`):

```
galaxy: int32 not null
x, y:   float32 not null
category: uint8   -- GZ10 morphology, null where unlabelled
```

- **`mean_points`**: one row per galaxy, from its mean embedding. Small, and loaded on first paint.
- **`full_points`**: one row per embedding, every modality in the same space; loaded only when the user asks for it. Written survey by survey, so its galaxy column is not monotonic.

Training logs to Weights & Biases under `WANDB_MODE`. `modal_app.py` sets it to `offline`, so runs are written to the volume and not uploaded (for now).

## `generate_pairs`

One row per galaxy, pairing AION's embedding of the galaxy with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)'s (`GEMMA`) embedding of its Legacy Survey image.

- **AION**: the mean of the galaxy's `encoded` embeddings over every token of every survey.
- **EmbeddingGemma**: its sentence-transformers embedding, L2-normalised and at full width, under the `Document` prompt, of the galaxy's `rgb` Legacy Survey cutout as the dataset stores it, uncropped.

Every image goes to one `encode` call, which batches internally:

```
galaxy: int32
aion:   fixed_size_list<float32, 768>
gemma:  fixed_size_list<float32, 768>
```

`GEMMA_DIM` must match the model's width: 768.

## `generate_alignment`

Fits two maps from a galaxy's AION embedding to its EmbeddingGemma embedding: least squares with a bias column, solved in closed form, and an MLP (`768 → HIDDEN (2048) → 768`, GELU, output L2-normalised) trained with Adam on 1 − cosine for `EPOCHS` (50). `VALIDATION` (30%) of galaxies are held out of both. Each epoch logs the training loss and, on the held-out rows, the mean cosine and `recall@10`, to Weights & Biases as `generate_projections` does; the least-squares map's two figures go to the run summary, and the run's config records `GEMMA` as `target`. `recall@10` is the share of held-out galaxies whose own EmbeddingGemma embedding is among the 10 nearest to their prediction, out of every held-out galaxy's.

**`alignment`** is a `torch.save` of:

```
linear: Tensor      -- (769, GEMMA_DIM), the last row the bias
mlp:    state_dict  -- AlignmentMap weights
```

## `generate_aion_gemma_space`

Applies the MLP in `alignment` to every galaxy's `pairs` AION embedding. **`aion_gemma_space`** is an `np.save` of:

```
float32 (galaxies, GEMMA_DIM)  -- in galaxy order, L2-normalised
```
