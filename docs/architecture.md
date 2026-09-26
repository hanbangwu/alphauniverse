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

| Role              | Shape                                      | Who reads it                                             |
| ----------------- | ------------------------------------------ | -------------------------------------------------------- |
| `encoded`         | one row per galaxy, embeddings per survey  | index build, projections, download                       |
| `codebook`        | same, the encoder's input embeddings       | nothing at serve time                                    |
| `tokens`          | same, token ids                            | startup, `/similarity`, the token and coverage endpoints |
| `encoded_index`   | faiss IVF over patches and spectral tokens | `/similarity`                                            |
| `cutouts`         | one PNG per galaxy, in galaxy order        | `/galaxies/{g}/image.png`                                |
| `spectra`         | one spectrum per matched survey, same      | `/galaxies/{g}/spectra/{s}`                              |
| `mean_points`     | one 2-d point per galaxy                   | `/meta`, the projection view                             |
| `full_points`     | one 2-d point per embedding                | the projection view                                      |
| `parametric_umap` | the trained projector's weights            | nothing at serve time                                    |

`docs/pipeline.md` has the schemas.

## The serving app

Eight endpoints, all `GET`; `/artifacts/{role}` also answers `HEAD` for range-request clients.

A request whose `If-None-Match` matches a response's ETag gets `304 Not Modified` with no body. An artifact's ETag is the one Starlette derives from the file's size and modification time, and the route compares it before reading the file. Every other successful response's ETag is an MD5 hash of its body, added by a middleware, so the server still builds the response and saves only the transfer. Every response carries `Cache-Control: no-cache`, so a browser revalidates a stored response before each reuse.

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

A query is one or more image patches and spectral spans of one galaxy. The answer is the query galaxy followed by up to `matches` other galaxies, ranked, each with a per-patch score map and, where it has a spectrum, a per-span score map; the frontend draws both.

The search reads vectors back from the index by id, using the layout in `docs/pipeline.md`:

1. The query's vectors are averaged and normalised into one direction.
2. The index returns the `PROBE` (2048) vectors nearest that direction, probing `NPROBE` (64) of its lists. The galaxies they belong to, in order of first appearance and without the query galaxy, are the candidates, cut to `matches`.
3. Every patch and span of the query galaxy and each candidate is scored by its cosine with the direction, giving the score maps.
4. A galaxy's score is its best token score over both maps. The candidates are sorted by it, after the query galaxy.

## The frontend

SvelteKit, Svelte 5 runes, one page in three resizable panes.

```
+page.svelte
├── LeftPanel        point set toggle, morphology filter, download
├── ProjectionView   embedding-atlas over a DuckDB-WASM table
└── RightPanel       selected galaxy: image or spectrum, morphology, crossmatches
    └── PatchSimilarity   dialog: query patches and spans, match count, Search, ranked matches
```

App-wide state lives in plain classes under `src/lib/state/`, held in a `runed` context and reached through the getters in `app.svelte.ts`; the similarity dialog keeps its own state beside its components in `similarity.svelte.ts`, including the last search submitted. The selected patches and spans and the match count are a draft: `/similarity` runs only when Search, or Enter in the count box, submits them, and a newly selected galaxy starts with no results. While a search runs, the previous results stay on screen, dimmed. `Field<T>` and its subclasses wrap a `$state` value with normalisation (sorting and deduping index lists), so the components never validate anything themselves. The match count is kept as the text typed into its number box, and `SearchState` checks it against bounds that are not typed by hand: `state/schema.ts` reads them out of the generated zod schema, so they come from the API contract.

`+layout.server.ts` fetches `/meta` during SSR. After that, two data paths, deliberately separate:

- **Row-level data** (tokens, coverage, spectra, similarity) goes through the generated client into TanStack Query. It is cached forever, except similarity, which goes stale after 5 minutes.
- **The point sets** skip the JSON endpoints. DuckDB-WASM reads the parquet artifact from `/artifacts/{role}` into a table, and Mosaic pushes the morphology filter into SQL so filtering the projection never round-trips to the server.

Patch grids are Apache ECharts custom series, one rect per patch on a 24×24 value grid, with selection and hover outlines drawn as rect strokes. The image token grid in the similarity dialog sits over the image at `tokenAlpha` opacity, 0.3 or 0.8 when selected, and the spectrum's token areas sit over the line at the same opacity. Each grid is its own ECharts instance. A match row holds the galaxy's image, its score grid, a mask grid while the Mask switch is on, and a spectrum chart where it has a spectrum; a match list is 32 rows by default and up to 128.

Spectra are Apache ECharts line charts. A galaxy's spectrum comes from the first of its matched spectrum surveys, DESI before SDSS. The interactive chart in the similarity dialog shades one area per spectrum token, laid out from `spectrum` in `/meta` and coloured like the token grid, and holds the selected spans in `view.spans`, which join the selected patches in the query; it zooms on the wheel and pans on drag because 272 spans do not fit a panel a few hundred pixels wide.

## Deployment

`main` deploys to Modal through `.github/workflows/ci-cd.yml`, except on pushes that change only docs or instructions. `docs/testing.md` lists the jobs. The frontend deploys separately on Vercel.
