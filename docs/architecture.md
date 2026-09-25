# Architecture

The system has two halves, connected only by a set of files on disk.

The build pipeline runs on Modal, generating artifacts. The app reads those artifacts and serves the platform.

```
hanbangwu/alphauniverse-cosmos (Hugging Face)
├── generate_embeddings       GPU ─▶ encoded · codebook · tokens
│   ├── generate_index        CPU ─▶ encoded_index
│   └── generate_projections  GPU ─▶ mean_points · full_points · parametric_umap
├── generate_cutouts          CPU ─▶ cutouts
└── generate_spectra          CPU ─▶ spectra

                every artifact, on the Modal volume at /cache
                                      │
                                      ▼
               app/main.py  (FastAPI, one Modal container)
                                      │
         ┌───────────────┬────────────┴──────┬──────────────────────┐
         │ /meta         │ JSON · Arrow ·    │ PNG                  │ parquet from
         │               │ raw uint32        │                      │ /artifacts/{role}
         ▼               ▼                   ▼                      ▼
    SSR layout    TanStack Query cache     <img>               DuckDB-WASM
         └───────────────┴─────────┬─────────┴──────────────────────┘
                                   ▼
                          SvelteKit single page
```

## The artifacts

| Role              | Shape                                      | Who reads it                          |
| ----------------- | ------------------------------------------ | ------------------------------------- |
| `encoded`         | one row per galaxy, embeddings per survey  | index build, projections, download    |
| `codebook`        | same, raw codebook vectors                 | nothing at serve time                 |
| `tokens`          | same, token ids                            | the `tokens` and `coverage` endpoints |
| `encoded_index`   | faiss IVF over patches and spectral tokens | `/similarity`                         |
| `cutouts`         | one PNG per galaxy, in galaxy order        | `/galaxies/{g}/image.png`             |
| `spectra`         | one spectrum per matched survey, same      | `/galaxies/{g}/spectra/{s}`           |
| `mean_points`     | one 2-d point per galaxy                   | `/meta`, the projection view          |
| `full_points`     | one 2-d point per embedding                | the projection view                   |
| `parametric_umap` | the trained projector's weights            | nothing at serve time                 |

`docs/pipeline.md` has the schemas.

## The serving app

Eight endpoints, all `GET`; `/artifacts/{role}` also answers `HEAD` for range-request clients.

| Endpoint                           | Returns      |
| ---------------------------------- | ------------ |
| `/meta`                            | JSON         |
| `/artifacts/{role}`                | the file     |
| `/galaxies/{g}/image.png`          | PNG          |
| `/galaxies/{g}/tokens`             | raw `uint32` |
| `/galaxies/{g}/spectra/{s}`        | Arrow IPC    |
| `/galaxies/{g}/spectra/{s}/tokens` | raw `uint32` |
| `/galaxies/{g}/coverage`           | JSON         |
| `/similarity`                      | Arrow IPC    |

## Similarity search

A query is one or more image patches and spectral spans of one galaxy, averaged into one direction. The answer is a ranked list of galaxies, each with a per-patch score map and, where it has a spectrum, a per-span score map; the frontend draws both. `app/search.py` states the method and the index layout it depends on.

## The frontend

SvelteKit, Svelte 5 runes, one page in three resizable panes.

```
+page.svelte
├── LeftPanel        point set toggle, morphology filter, download
├── ProjectionView   embedding-atlas over a DuckDB-WASM table
└── RightPanel       selected galaxy: image or spectrum, morphology, crossmatches
    └── PatchSimilarity   dialog: query patches, spectrum spans, ranked matches
```

App-wide state lives in plain classes under `src/lib/state/`, held in a `runed` context and reached through the getters in `app.svelte.ts`; the similarity dialog keeps its own state beside its components in `similarity.svelte.ts`. `Field<T>` and its subclasses wrap a `$state` value with normalisation (clamping numbers, sorting and deduping index lists), so the components never validate anything themselves. Numeric bounds are not typed by hand: `state/schema.ts` reads them out of the generated zod schema, so the slider ranges come from the API contract.

`+layout.server.ts` fetches `/meta` during SSR. After that, two data paths, deliberately separate:

- **Row-level data** (tokens, coverage, spectra, similarity) goes through the generated client into TanStack Query. It is cached forever, except similarity, which goes stale after 5 minutes.
- **The point sets** skip the JSON endpoints. DuckDB-WASM reads the parquet artifact from `/artifacts/{role}` into a table, and Mosaic pushes the morphology filter into SQL so filtering the projection never round-trips to the server.

Patch grids are Apache ECharts custom series, one rect per patch on a 24×24 value grid, with selection and hover outlines drawn as rect strokes. The image token grid in the similarity dialog sits over the image at `tokenAlpha` opacity, 0.3 or 0.8 when selected, and the spectrum's token areas sit over the line at the same opacity. Each grid is its own ECharts instance. A match row holds the galaxy's image, its score grid, a mask grid while the Mask switch is on, and a spectrum chart where it has a spectrum; a match list is 32 rows by default and up to 128.

Spectra are Apache ECharts line charts. A galaxy's spectrum comes from the first of its matched spectrum surveys, DESI before SDSS. The interactive chart in the similarity dialog shades one area per spectrum token, laid out from `spectrum` in `/meta` and coloured like the token grid, and holds the selected spans in `view.spans`, which join the selected patches in the query; it zooms on the wheel and pans on drag because 272 spans do not fit a panel a few hundred pixels wide.

## Deployment

`main` deploys to Modal through `.github/workflows/ci-cd.yml`. `docs/testing.md` lists the jobs. The frontend deploys separately on Vercel.
