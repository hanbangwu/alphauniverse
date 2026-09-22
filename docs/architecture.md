# Architecture

The system has two halves, connected only by a set of files on disk.

The build pipeline runs on Modal, generating artifacts. The app reads those artifacts and serves the platform.

```
 hanbangwu/alphauniverse-cosmos (Hugging Face)
              │                                  │
              │  generate_embeddings  GPU, ~3h   │  generate_cutouts · generate_spectra  CPU
              ▼                                  ▼
   encoded · codebook · tokens ─────┐      cutouts · spectra
              │                     │            │
              │  generate_index     │  generate_projections  GPU
              │    CPU              │            │
              ▼                     ▼            │
      encoded_index.faiss   parametric_umap ·    │
                            mean_points ·        │
                            full_points          │
              │                     │            │
              └──────────┬──────────┴────────────┘
                         ▼
                 app/main.py  (FastAPI, one Modal container)
                         │
             ┌───────────┴────────────┐
             │ JSON + PNG + Arrow     │ parquet artifacts
             ▼                        ▼
    TanStack Query cache      DuckDB-WASM in the browser
             └───────────┬────────────┘
                         ▼
                SvelteKit single page
```

## The artifacts

| Role              | Shape                                      | Who reads it                        |
| ----------------- | ------------------------------------------ | ----------------------------------- |
| `encoded`         | one row per galaxy, embeddings per survey  | index build, projections, download  |
| `codebook`        | same, raw codebook vectors                 | nothing at serve time               |
| `tokens`          | same, token ids                            | `/galaxies/{g}/tokens`, `/coverage` |
| `encoded_index`   | faiss IVF over patches and spectral tokens | `/similarity`                       |
| `cutouts`         | one PNG per galaxy, in galaxy order        | `/galaxies/{g}/image.png`           |
| `spectra`         | one spectrum per matched survey, same      | `/galaxies/{g}/spectra/{s}`         |
| `mean_points`     | one 2-d point per galaxy                   | `/meta`, the projection view        |
| `full_points`     | one 2-d point per embedding                | the projection view                 |
| `parametric_umap` | the trained projector's weights            | nothing at serve time               |

`docs/pipeline.md` has the schemas.

## The serving app

Eight endpoints, all `GET`.

| Endpoint                           | Returns      | Cost per request                      |
| ---------------------------------- | ------------ | ------------------------------------- |
| `/meta`                            | JSON         | cached after the first call           |
| `/artifacts/{role}`                | the file     | disk read, streamed; up to tens of GB |
| `/galaxies/{g}/image.png`          | PNG          | array lookup into `cutouts`           |
| `/galaxies/{g}/tokens`             | raw `uint32` | one filtered parquet read             |
| `/galaxies/{g}/spectra/{s}`        | Arrow IPC    | array lookup into `spectra`           |
| `/galaxies/{g}/spectra/{s}/tokens` | raw `uint32` | one filtered parquet read             |
| `/galaxies/{g}/coverage`           | JSON         | one filtered parquet read             |
| `/similarity`                      | Arrow IPC    | one ANN search + candidate rescoring  |

## Similarity search

A query is one or more image patches and spectral spans of one galaxy, averaged into one direction. The answer is a ranked list of galaxies, each with a per-patch score map and a per-span score map; the frontend draws the first and ignores the second. `app/search.py` states the method and the index layout it depends on.

## The frontend

SvelteKit, Svelte 5 runes, one page in three resizable panes.

```
+page.svelte
├── LeftPanel        point set toggle, morphology filter, download
├── ProjectionView   embedding-atlas over a DuckDB-WASM table
└── RightPanel       selected galaxy: cutout, tokens, spectrum, coverage
    └── PatchSimilarity   dialog: query patches, spectrum spans, ranked matches
```

State lives in plain classes under `src/lib/state/`, held in a `runed` context and reached with `getApp()`, `getView()`, `getSearch()`, `getMask()`, `getFilters()`. `Field<T>` and its subclasses wrap a `$state` value with normalisation (clamping numbers, sorting and deduping index lists), so the components never validate anything themselves. Numeric bounds are not typed by hand: `state/schema.ts` reads them out of the generated zod schema, so the slider ranges come from the API contract.

Two data paths, deliberately separate:

- **Row-level data** (`meta`, tokens, coverage, spectra, similarity) goes through the generated client into TanStack Query. Everything keyed by revision is cached forever; similarity has a 5-minute stale time.
- **The point sets** never pass through the app. DuckDB-WASM fetches the parquet artifact directly over range requests, and Mosaic pushes the morphology filter into SQL so filtering the projection never round-trips to the server.

Patch grids render to `<canvas>` at native resolution, one `ImageData` of 24×24 pixels scaled up with `image-rendering: pixelated`, plus a second canvas for selection and hover outlines. `IsInViewport` gates the redraw, which matters because a full match list is ~32 rows of three grids each.

Spectra are Apache ECharts line charts. A galaxy's spectrum comes from the first of its matched spectrum surveys, DESI before SDSS. The interactive chart in the similarity dialog shades one area per spectrum token, laid out from `spectrum` in `/meta` and coloured like the token grid, and holds the selected spans in `view.spans`, which join the selected patches in the query; it zooms on the wheel and pans on drag because 272 spans do not fit a panel a few hundred pixels wide.

## Deployment

`main` deploys to Modal through `.github/workflows/ci-cd.yml`. `docs/testing.md` lists the jobs. The frontend deploys separately on Vercel. `docs/performance.md` measures the relevant performance metrics.
