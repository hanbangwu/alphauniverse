# Architecture

Two halves, connected only by files on disk: the build pipeline, Modal jobs that write artifacts, and the app, which reads them and serves the platform.

```
hanbangwu/alphauniverse-cosmos (Hugging Face)
└── generate_embeddings       GPU ─▶ encoded · codebook · tokens
    ├── generate_index        CPU ─▶ search_index
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
    SSR layout    TanStack Query cache     <img>               DuckDB-WASM
         └───────────────┴─────────┬─────────┴──────────────────────┘
                                   ▼
                          SvelteKit single page
```

## The artifacts

| Role               | Shape                                         | Who reads it                                                            |
| ------------------ | --------------------------------------------- | ----------------------------------------------------------------------- |
| `encoded`          | one row per galaxy, embeddings per survey     | index build, projections, `/downloads/{role}`                           |
| `codebook`         | same, the encoder's input embeddings          | `/downloads/{role}`                                                     |
| `tokens`           | same, token ids                               | startup, `/search`, the token and galaxy endpoints, `/downloads/{role}` |
| `search_index`     | faiss IVF over patches, spans and scalars     | `/search`                                                               |
| `mean_points`      | one 2-d point per galaxy                      | `/meta`, `/projections/mean`                                            |
| `full_points`      | one 2-d point per embedding                   | `/projections/full`                                                     |
| `parametric_umap`  | the trained projector's weights               | nothing at serve time                                                   |
| `pairs`            | AION and EmbeddingGemma embeddings per galaxy | `generate_alignment`, `text_search_quality`                             |
| `alignment`        | the AION to EmbeddingGemma maps' weights      | `generate_aion_gemma_space`, `text_search_quality`                      |
| `aion_gemma_space` | each galaxy's vector in EmbeddingGemma space  | `/search/text`                                                          |

`docs/pipeline.md` has the schemas. `/meta` counts `mean_points.category`: one count per GZ10 class, unlabelled galaxies last. `/galaxy/{g}/image`, `/galaxy/{g}/spectrum` and `/galaxy/{g}/table` read the cached dataset instead; `/galaxy/{g}/spectrum` smooths the flux with astropy for display only (Gaussian, `SPECTRUM_SMOOTHING_SIGMA` pixels, masked pixels stay NaN), and search reads the unsmoothed flux. `/galaxy/{g}/table` returns every numeric and boolean column whose name ends in one of the six catalogue suffixes, null where that catalogue has no match; each of the 25 scalars AION encodes (12 Legacy Survey, then 13 HSC, `SCALAR_SURVEYS` in `app/config.py`) carries its index.

## The serving app

Eleven endpoints, all `GET`; `/projections/{projection}` and `/downloads/{role}` also answer `HEAD`, for range-request clients and size checks. `/downloads/{role}` serves only `encoded`, `codebook` and `tokens`.

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

## Similarity search

A query is one or more image patches, spectral spans and encoded scalars of one galaxy. An HSC scalar needs an HSC match. The answer is the query galaxy, then `matches` other galaxies, or every other galaxy if the dataset holds fewer, ranked, each with a per-patch score map, a per-span score map where it has a spectrum, and a per-scalar score map, NaN for the HSC scalars where it has no HSC match.

The search reads vectors back from the index by id, in the layout `docs/pipeline.md` gives:

1. The query's vectors are averaged and normalised into one direction.
2. The index returns the `PROBE` (2048) vectors nearest that direction, probing `NPROBE` (64) of its lists. The galaxies they belong to, in order of first appearance and without the query galaxy, are the candidates, cut to `matches`. If they hold fewer galaxies, the search runs again with both numbers doubled, until there are enough or it has probed every list for every vector. The lists probed are set per search, so the index that concurrent requests share is not changed. faiss scans them on its threads and starts no prefetch threads.
3. Every patch, span and scalar of the query galaxy and each candidate is scored by its cosine with the direction, giving the score maps.
4. A galaxy's score is its best token score over its maps. The candidates are sorted by it, after the query galaxy.

## Text search

A query is free text and a match count. `/search/text` embeds the text with EmbeddingGemma in the API process, under the `SearchQuery` prompt; the model loads on the first text search, from the Hugging Face cache, which `generate_pairs` fills on Modal. The answer is the `matches` galaxies whose `aion_gemma_space` vectors have the highest cosine with it, ranked. `aion_gemma_space` is loaded on the first text search.

In the left panel, the text box submits on Search or Enter; the results are a grid of thumbnails with galaxy id and score, and clicking one selects the galaxy.

## The frontend

SvelteKit, Svelte 5 runes, one page in three resizable panes.

```
+page.svelte
├── LeftPanel        point set toggle, text search, morphology filter, download
├── ProjectionView   embedding-atlas over a DuckDB-WASM table
│   └── GalaxyTooltip     hovered or selected galaxy: image, id, morphology
└── RightPanel       selected galaxy: image or spectrum, morphology, crossmatches
    └── PatchSimilarity   dialog: query patches, spans and scalars, match count, Search, ranked matches
```

App-wide state is plain classes under `src/lib/state/`, held in a `runed` context and reached through the getters in `app.svelte.ts`. The similarity dialog keeps its own state, including the last search submitted, beside its components in `similarity.svelte.ts`.

- The selected patches and spans and the match count are a draft: `/search` runs only when Search, or Enter in the count box, submits them. A newly selected galaxy starts with no results. While a search runs, the previous results stay on screen, dimmed.
- `Field<T>` and its subclasses wrap a `$state` value with normalisation (sorting and deduping index lists), so components never validate.
- The match count is kept as the typed text. `SearchState` checks it against bounds that `state/schema.ts` reads from the generated zod schema, so they come from the API contract.

`+layout.server.ts` fetches `/meta` during SSR. After that, two data paths, deliberately separate:

- **Row-level data** (tokens, coverage, spectra, similarity) goes through the generated client into TanStack Query. It is cached forever, except similarity, which goes stale after 5 minutes.
- **The point sets** skip the JSON endpoints. DuckDB-WASM reads the parquet artifact from `/projections/{projection}` into a table, and Mosaic pushes the morphology filter into SQL so filtering the projection never round-trips to the server.

Patch grids are Apache ECharts custom series, one rect per patch on a 24×24 value grid, with selection and hover outlines drawn as rect strokes. Each grid is its own ECharts instance. In the similarity dialog the image sits over the image token grid, blended with `mix-blend-screen`. Before a search the grid is opaque and the image at `tokenAlpha` opacity, 0.3; while the pointer is over the panel the image is opaque and the grid takes `tokenAlpha` opacity, 0.3 or 0.8 when selected. After a search the grid shows the score map and both are opaque. Before a search the spectrum's token areas sit over the line at `tokenAlpha` opacity. A match row holds the galaxy's image, its score grid, a mask grid while the Mask switch beside the query galaxy's Image Tokens title is on, and a spectrum chart where it has a spectrum. The Invert switch sits beside the query galaxy's Image Mask title, and the threshold slider under its grid. Panel titles stay the same before and after a search. A match list is 32 rows by default and up to 128.

Spectra are Apache ECharts line charts. The app serves and shows only DESI spectra; a galaxy without one shows no spectrum. The dialog's interactive chart shades one area per spectrum token, laid out from `spectrum_origin` and `spectrum_width` in `/meta` and coloured like the token grid. It holds the selected spans in `view.spans`, which join the selected patches in the query. It zooms on the wheel and pans on drag, since 272 spans do not fit a panel a few hundred pixels wide.

Right of the image panels, the dialog's Tabular Data table lists `/galaxy/{g}/table` grouped by catalogue. Each encoded scalar has a checkbox; the checked ones are `view.scalars` and join the query. An encoded scalar without a token, because its catalogue has no match, has a disabled checkbox and the not-allowed cursor. Before a search an encoded row's background takes its token's colour, ranked as the token grid's are, at `tokenAlpha` opacity; after a search it takes the row's score on viridis over the 25 scores. A checked row has a 1 px `SELECTED` outline, as a selected span does.

## Deployment

`main` deploys to Modal through `.github/workflows/ci-cd.yml`, except on pushes that change only docs or instructions. `docs/testing.md` lists the jobs. The frontend deploys separately on Vercel.
