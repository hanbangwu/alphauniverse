# Architecture

Two halves, connected only by files on disk: the build pipeline, Modal jobs that write artifacts, and the app, which reads them and serves the platform.

```
hanbangwu/alphauniverse-cosmos (Hugging Face)
└── generate_embeddings       GPU ─▶ encoded · codebook · tokens
    ├── generate_index        CPU ─▶ search_index
    ├── generate_predictions  GPU ─▶ predictions · prediction_basis
    ├── generate_projections  GPU ─▶ mean_points · full_points · parametric_umap
    └── generate_pairs        GPU ─▶ pairs
        └── generate_alignment GPU ─▶ alignment
            └── generate_aion_gemma_space GPU ─▶ aion_gemma_space

  every artifact, the cached dataset and EmbeddingGemma, on the Modal volume at /cache
                                      │
                                      ▼
               app/main.py  (FastAPI, one Modal container)
                                      │
         ┌───────────────┬────────────┴──────┬──────────────────────┐
         │ /meta         │ JSON · Arrow ·    │ PNG                  │ parquet from
         │               │ raw uint32        │                      │ /projections/{p}
         ▼               ▼                   ▼                      ▼
    layout        TanStack Query cache     <img>               DuckDB-WASM
         └───────────────┴─────────┬─────────┴──────────────────────┘
                                   ▼
                          SvelteKit single page
```

## The artifacts

| Role               | Shape                                                         | Who reads it                                                             |
| ------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `encoded`          | one row per galaxy, embeddings per survey                     | index build, projections, `/downloads/{role}`                            |
| `codebook`         | same, the encoder's input embeddings                          | `/downloads/{role}`                                                      |
| `tokens`           | same, token ids                                               | startup, `/search`, token and galaxy endpoints, downloads, predictions   |
| `search_index`     | faiss IVF over image tokens, spectrum tokens and table values | `search_quality`, `search_performance`, `pql_quality` (#219)             |
| `mean_points`      | one 2-d point per galaxy                                      | `/meta`, `/projections/mean`                                             |
| `full_points`      | one 2-d point per embedding                                   | `/projections/full`                                                      |
| `parametric_umap`  | the trained projector's weights                               | nothing at serve time                                                    |
| `pairs`            | AION and EmbeddingGemma embeddings per galaxy                 | `generate_alignment`, `generate_aion_gemma_space`, `text_search_quality` |
| `alignment`        | the AION to EmbeddingGemma maps' weights                      | `generate_aion_gemma_space`, `text_search_quality`                       |
| `aion_gemma_space` | each galaxy's vector in EmbeddingGemma space                  | `/search/text`                                                           |
| `predictions`      | AION's predicted codes at every slot                          | startup, `/search`, `pql_quality`                                        |
| `prediction_basis` | the spectrum token predictions' PCA basis per survey          | startup, `/search`, `pql_quality`                                        |

`docs/pipeline.md` has the schemas. `/meta` counts `mean_points.category`: one count per GZ10 class, unlabelled galaxies last. `/galaxy/{g}/image`, `/galaxy/{g}/spectrum` and `/galaxy/{g}/table` read the cached dataset instead; `/galaxy/{g}/spectrum` smooths the flux with astropy for display only (Gaussian, `SPECTRUM_SMOOTHING_SIGMA` pixels, masked pixels stay NaN), and search reads the unsmoothed flux. `/galaxy/{g}/table` returns every numeric and boolean column whose name ends in one of the six catalogue suffixes, null where that catalogue has no match; the `Z` row AION was given carries index 0, the redshift, and each of the 25 table values AION encodes carries its index: 1 to 12 Legacy Survey, then 13 to 25 HSC (`TABLE_VALUE_SURVEYS` in `app/config.py`). The DESI and SDSS redshift rows (`Z`, its error, its flag) come first, in the `redshift` section and named with their survey; every other row's section is its catalogue. Any other measured `Z` row has `excluded`, the reason: flagged by its survey, above AION's limit of 6, or not the one AION takes.

## The serving app

Eleven endpoints, all `GET`; `/projections/{projection}` and `/downloads/{role}` also answer `HEAD`, for range-request clients and size checks. `/downloads/{role}` serves only `encoded` and `tokens`, as Arrow IPC files, and `codebook`, as Parquet.

A request whose `If-None-Match` matches the ETag gets `304 Not Modified` with no body. An artifact's ETag is Starlette's, from the file's size and modification time, compared before the file is read. Every other successful response's ETag is an MD5 of its body, added by the route class every endpoint uses: the server still builds the response and saves only the transfer. Successful responses and 304s carry `Cache-Control: no-cache`, so a browser revalidates before each reuse.

| Endpoint                      | Returns      |
| ----------------------------- | ------------ |
| `/meta`                       | JSON         |
| `/projections/{projection}`   | the file     |
| `/downloads/{role}`           | the file     |
| `/galaxy/{g}`                 | JSON         |
| `/galaxy/{g}/image`           | PNG          |
| `/galaxy/{g}/image/tokens`    | raw `uint32` |
| `/galaxy/{g}/spectrum`        | Arrow IPC    |
| `/galaxy/{g}/spectrum/tokens` | raw `uint32` |
| `/galaxy/{g}/table`           | JSON         |
| `/search`                     | Arrow IPC    |
| `/search/text`                | JSON         |

`/search` takes `galaxy`, the selected slots `ls_image`, `hsc_image`, `desi_spectrum`, `sdss_spectrum` and `table_values`, each repeated once per index, and `matches` (1–128, default 32). `/search/text` takes `text` (1–500 characters) and `matches` (1–128, default 32).

## Similarity search

A query is one galaxy and a selection from it: image tokens of its Legacy Survey or HSC image, spectrum tokens of its DESI or SDSS spectrum, and table values, numbered as `/galaxy/{g}/table` numbers them (0 the redshift, 1 to 12 Legacy Survey, 13 to 25 HSC). Every selected mode must be one the galaxy has: the HSC image and HSC table values need an HSC match, a spectrum's tokens that spectrum, the redshift a redshift token.

`app/pql.py` scores by predictive query likelihood (#199) from `predictions` and `prediction_basis`, which `lifespan` loads, by an exact scan of every galaxy:

1. The selection splits into modes: each image survey, each spectrum survey, each catalogue's table values and the redshift.
2. At each selected slot, a galaxy's overlap with the query is Σ_t p_q(t) p_g(t) over their predicted distributions. Image tokens use the query's 64 stored codes and the galaxy's top `KEPT` (16), each with the rest of its mass spread evenly; spectrum tokens use the PCA coefficients, floored at 0.1/1,024; table values and the redshift take a log-sum-exp of the stored log-probabilities.
3. A mode's sum is the sum of its slots' log overlaps, in nats. One mode ranks by its sum; several rank by the mean of each mode's sum standardised over every galaxy.
4. The answer is the query galaxy's row, then the `matches` best other galaxies, or every other galaxy if the dataset holds fewer.

Each row carries its `score`, `{mode}_sum` for each selected mode (null otherwise), `has_hsc`, `has_desi`, `has_sdss` and `has_redshift`, and two kinds of map:

- **Aligned maps** (`ls_image`, `hsc_image`, `desi_spectrum`, `sdss_spectrum`, `table_values`), for every mode: at each slot, the log overlap of the galaxy's and the query galaxy's predictions at that slot. A mode's sum is its aligned map summed over the selected slots.
- **Selection maps** (`{mode}_selection`), for each selected image or spectrum mode, null otherwise: at each slot, the log overlap with the mean of the query's predictions over its selected slots of that mode.

Every map comes from predictions, so a galaxy without a mode has maps for it too; the `has_` flags say which modes it observed. The dialog shows a match's Legacy Survey image map and DESI spectrum map: the selection map when that mode is selected, the aligned map otherwise.

## Text search

A query is free text and a match count. `/search/text` embeds the text with EmbeddingGemma in the API process, under the `SearchQuery` prompt. The first text search imports sentence-transformers and torch and loads the model from the Hugging Face cache, which `generate_pairs` fills on Modal. The answer is the `matches` galaxies whose `aion_gemma_space` vectors have the highest cosine with it, ranked. `aion_gemma_space` is loaded on the first text search.

The Text Search tab beside the projection submits on Search or Enter. Each result row shows the galaxy's thumbnail, id, score and morphology; clicking it selects the galaxy.

## The frontend

SvelteKit, Svelte 5 runes, one page in three resizable panes.

```
+page.svelte
├── LeftPanel        point set toggle, morphology filter, download
├── Tabs
│   ├── ProjectionView    Parametric UMAP: embedding-atlas over a DuckDB-WASM table
│   │   └── GalaxyTooltip     hovered or selected galaxy: image, id, morphology
│   └── TextSearch        Text Search: text box, ranked galaxies
└── RightPanel       selected galaxy: image or spectrum, morphology, crossmatches
    └── ImageTokenSimilarity   dialog: query image tokens, spectrum tokens and table values, match count, Search, ranked matches
```

App-wide state is plain classes under `src/lib/state/`, held in a `runed` context and reached through the getters in `app.svelte.ts`. The similarity dialog keeps its own state, including the last search submitted, beside its components in `similarity.svelte.ts`.

- The selected image tokens, spectrum tokens and table values and the match count are a draft: `/search` runs only when Search, or Enter in the count box, submits them. A newly selected galaxy starts with no results. While a search runs, the previous results stay on screen, dimmed.
- `Field<T>` and its subclasses wrap a `$state` value with normalisation (sorting and deduping index lists), so components never validate.
- The match count is kept as the typed text. `SearchState` checks it against bounds that `state/schema.ts` reads from `/search`'s parameters in `openapi.json`, so they come from the API contract.

The layout fetches `/meta` in the browser through TanStack Query, cached forever, and shows a spinner until it returns. DuckDB-WASM starts loading the `mean` points at the same time. Data then takes two paths, deliberately separate:

- **Row-level data** (tokens, coverage, table, spectra, similarity, text search) goes through the generated client into TanStack Query. Similarity and text search results never go stale and leave the cache 5 minutes after nothing reads them; everything else stays for the page session.
- **The point sets** skip the JSON endpoints. DuckDB-WASM reads the parquet artifact from `/projections/{projection}` into a table, and Mosaic pushes the morphology filter into SQL so filtering the projection never round-trips to the server.

Image token grids are Apache ECharts custom series, one rect per image token on a 24×24 value grid, with selection and hover outlines drawn as rect strokes. Each grid is its own ECharts instance. In the similarity dialog the image sits over the image token grid, blended with `mix-blend-screen`. Before a search the grid is opaque and the image at `tokenAlpha` opacity, 0.3; while the pointer is over the panel the image is opaque and the grid takes `tokenAlpha` opacity, 0.3 or 0.8 when selected. After a search the grid shows the score map and both are opaque. Before a search the spectrum's token areas sit over the line at `tokenAlpha` opacity. A match row holds the galaxy's image, its score grid, a mask grid while the Mask switch beside the query galaxy's Image Tokens title is on, and a spectrum chart where it has a spectrum. The Invert switch sits beside the query galaxy's Image Mask title, and the threshold slider under its grid. Panel titles stay the same before and after a search. A match list is 32 rows by default and up to 128.

Spectra are Apache ECharts line charts. The app serves and shows only DESI spectra; a galaxy without one shows no spectrum. The dialog's interactive chart shades one area per spectrum token, laid out from `spectrum_origin` and `spectrum_width` in `/meta` and coloured like the token grid. It holds the selected spectrum tokens in `view.spectrum_tokens`, which join the selected image tokens in the query. It zooms on the wheel and pans on drag, since 272 spectrum tokens do not fit a panel a few hundred pixels wide.

Right of the image panels, the dialog's Tabular Data table lists `/galaxy/{g}/table` grouped by section, Redshift first. Each encoded table value has a checkbox; the checked ones are `view.table_values` and join the query. An encoded table value without a token, because its catalogue has no match, has a disabled checkbox and the not-allowed cursor; a row with `excluded` has a disabled checkbox and is grey and italic, with the reason under its name. Before a search an encoded row's background takes its token's colour, ranked as the token grid's are, at `tokenAlpha` opacity; after a search it takes the row's score on viridis over the 26 scores. A checked row has a 1 px `SELECTED` outline, as a selected spectrum token does.

## Deployment

`main` deploys to Modal through `.github/workflows/ci-cd.yml`, except on pushes that change only docs or instructions. `docs/testing.md` lists the jobs. The frontend deploys separately on Vercel.
