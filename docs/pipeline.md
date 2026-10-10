# Build pipeline

Seven Modal jobs; the README has the commands. Embeddings come first, then the index and the projections, which read `encoded`, and `generate_predictions`, which reads `tokens`. `generate_pairs` reads the dataset and `encoded`, `generate_alignment` reads `pairs`, and `generate_aion_gemma_space` reads `pairs` and `alignment`. All jobs share a volume mounted at `/cache`, and `ALPHAUNIVERSE_CACHE` points the app at it. Artifacts go under `$ALPHAUNIVERSE_CACHE/<author>/<name>/<revision>/`, so changing `DATASET_REVISION` switches trees rather than overwriting one.

Each job writes its artifacts in place, so a failed run leaves them incomplete; rerun the job. Every artifact holds one row per galaxy in dataset row order, and the app relies on that without checking it, so a build directory must come from one complete run of the pipeline.

Every job runs on the build image, which adds the `build` dependency group (AION, scikit-learn, torchvision, umap-learn, wandb). The serving image installs only the main dependencies, which include torch and sentence-transformers for text search.

## The dataset

`hanbangwu/alphauniverse-cosmos` is built by `scripts/alphauniverse_cosmos.py`: Legacy Survey DR10 south, cut to a √2° box on the COSMOS field, left-joined against five catalogues with LSDB and pushed to the Hub. The README lists the surveys and their token counts.

The script is outside the deployed pipeline and needs the `dataset` dependency group (lsdb, dask): run it with `uv run --group dataset python -m scripts.alphauniverse_cosmos`.

The serving app reads images and spectra from the dataset itself, from the copy `generate_embeddings` cached on the volume, with `HF_HUB_OFFLINE=1`. Offline, `datasets` ignores `DATASET_REVISION` and loads the most recently cached revision, so the volume must hold only the revision the artifacts were built from.

- **Image**: the anchor survey's `rgb` cutout, centre-cropped to `CROP_PIXELS` square and PNG-encoded on each request.
- **Spectrum**: wavelength in Ångström. Samples the survey pads with (wavelength at or below zero) are dropped, and samples it masks have NaN flux.

## `generate_embeddings`

For each galaxy: tokenise every modality it has, run all its tokens through the AION encoder in one pass, then split the output back apart by modality id. A spectrum's padding samples (wavelength at or below zero) are dropped before tokenising, as the serving app drops them. AION's one redshift token (`tok_z`) takes the DESI `Z` if it is usable, else the SDSS `Z`: usable means present, not NaN, unflagged (`ZWARN`, `ZWARNING`) and at most the trained codec's upper limit, 6 (`REDSHIFT_LIMIT`). The codec clamps a negative redshift to its first bin.

The model and every codec load from the latest commit of `polymathic-ai/aion-base`, so a push to that repository changes the next build.

Galaxies are encoded one at a time and written in batches of 128 (`encode.BATCH`). `encoded` and `tokens` are uncompressed Arrow IPC files, one record batch per batch, which every reader memory-maps; `codebook` is Parquet, one row group per batch.

Three stores are written, with the same columns:

```
galaxy: int32
ls, hsc, desi, sdss: list<fixed_size_list<float16, 768>>   -- null where unmatched
redshift:            list<fixed_size_list<float16, 768>>   -- one token, null without a usable redshift
gz10, provabgs:      bool
```

- **`encoded`**: what AION's `_encode` returns, the context its decoder reads. That is the encoder's output after `encoder_norm`, mapped by the linear `decoder_proj_context`, with each token's position and modality embeddings added back. Those added embeddings depend only on the token's modality and position, so every galaxy shares them. Because every modality is encoded together, a galaxy's spectrum tokens carry information from its image.
- **`codebook`**: the encoder's input embedding of each token, before position and modality embeddings are added or any context is mixed in, so it depends only on the token id and its modality.
- **`tokens`**: the token ids, so each survey cell is a `list<uint32>` instead of a list of embeddings.

Within an image cell the image tokens come first and the survey's table values follow. A spectrum cell leads with the codec's normalisation token, then holds one token per 25.6 Å from 3500 Å. AION resamples every spectrum onto 8704 pixels of 0.8 Å from 3500 Å and downsamples by 32, so a spectrum cell holds 273 tokens whatever survey it came from: the normalisation token and 272 spectrum tokens.

## `generate_index`

Builds `IVF{nlist},SQfp16` over one block per galaxy, in galaxy order: the anchor survey's 576 **image tokens**; then, if the galaxy has a spectrum, the 272 spectral tokens of its first matched spectrum survey, DESI before SDSS, with the normalisation token dropped; then the anchor survey's 12 **table values**; then, if the galaxy has an HSC match, HSC's 13 table values; then, if it has a redshift token, its **redshift**. HSC's image tokens are not indexed. Inner product is the metric and rows are L2-normalised first, so inner product is cosine similarity. A row that is not finite fails the build.

A vector's id is its position in that sequence: galaxy `g` starts at `588 g + 272 s + 13 h + r`, where `s`, `h` and `r` count the galaxies before it that have a spectrum, an HSC match and a redshift token. The index does not store this layout: `app/search.py` rebuilds it for the benchmark scripts from which galaxies have a spectrum, an HSC match and a redshift in `tokens`, so it holds only while `tokens` and `encoded` agree on that. One `generate_embeddings` run writes both.

## `generate_predictions`

For each galaxy: run every token it has through the AION encoder in one pass, with no truncation, then decode AION's distribution over codes at every slot, whether or not the galaxy has that mode: the redshift, the 576 image tokens and the table values of the Legacy Survey and HSC images, and spectrum tokens 1 to 272 of the DESI and SDSS spectra. Each mode's slots are split into blocks of 128, in an order drawn with `SEED`, and the decoder predicts every block of a galaxy in one call; a slot attends only to the slots of its own block, as in AION's default call of 128. `/search` scores from them (`docs/architecture.md`).

The job makes two passes over `tokens`. The first predicts only the spectrum tokens and accumulates, per spectrum survey, the sum and the sum of outer products of every spectrum token's probabilities; their covariance's top 256 eigenvectors and the mean form **`prediction_basis`**, an `np.savez` of:

```
desi_mean, sdss_mean:             float64 (1024,)
desi_directions, sdss_directions: float64 (256, 1024)
```

The second predicts every slot and writes **`predictions`**, an uncompressed Arrow IPC file meant to be memory-mapped, in batches of 256 rows:

```
galaxy:                    int32
redshift:                  fixed_size_list<float16, 1025>      -- the redshift's log-probabilities over AION's 1,025 `tok_z` codes
ls_codes, hsc_codes:       fixed_size_list<uint16, 576 × 64>   -- each image token's 64 most probable codes, most probable first
ls_log_probabilities, ...: fixed_size_list<float16, 576 × 64>  -- their log-probabilities
ls_tails, hsc_tails:       fixed_size_list<float32, 576>       -- log of each image token's remaining mass
ls_table_values:                fixed_size_list<float16, 12 × 1024> -- every table value's log-probabilities
hsc_table_values:               fixed_size_list<float16, 13 × 1024>
desi_coefficients, ...:    fixed_size_list<uint8, 272 × 256>   -- each spectrum token's probabilities projected on the basis
desi_offsets, ...:         fixed_size_list<float32, 272>       -- per spectrum token, coefficient = offset + code × step
desi_steps, ...:           fixed_size_list<float32, 272>
```

A spectrum token's probabilities are `mean + coefficients @ directions`. A row takes 496,390 B: 2,050 for the redshift, 149,760 per survey's image tokens, 24,576 and 26,624 for the table values, 71,808 per survey's spectrum tokens and 4 for `galaxy`. AION's `tok_z` has 1,025 codes where its codec emits 0 to 1023; all 1,025 are stored as AION gives them.

## `generate_projections`

Fits a parametric UMAP on a sample of embeddings and applies it to every one, in two passes over `encoded`, survey by survey. `SAMPLE` (5,000,000) embeddings are drawn uniformly from the whole store, and `train_test_split` holds out `VALIDATION` (30%) of them. The first pass accumulates each galaxy's mean embedding over all its tokens and writes each drawn embedding into one array, training rows then validation rows. The projector, an MLP from a normalised 768-d embedding to 2-d, trains on the training rows: UMAP's fuzzy simplicial set over the training rows weights the edges between neighbours, and each step draws edges by weight, pulls their endpoints together, and pushes each edge's first endpoint away from `NEGATIVES` (5) random rows. Each epoch logs `train/loss` and `validation/loss` to Weights & Biases when `WANDB_API_KEY` is set, and logs nothing otherwise; on Modal the job reads it from the `wandb-secret` secret; the validation loss is the same loss over the held-out rows' own fuzzy simplicial set, without gradients. The second pass projects every embedding.

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

## `generate_pairs`

One row per galaxy, pairing AION's embedding of the galaxy with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)'s (`GEMMA`) embedding of its Legacy Survey image.

- **AION**: the mean of the galaxy's `encoded` embeddings over every token of every survey.
- **EmbeddingGemma**: its sentence-transformers embedding, L2-normalised and at full width, under the `Document` prompt, of the galaxy's `rgb` Legacy Survey cutout as the dataset stores it, uncropped.

Images are decoded and passed to `encode` `BATCH` (1024) at a time:

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
