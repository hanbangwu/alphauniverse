# Architecture

Two halves that meet at a set of files.

The **build pipeline** runs offline on Modal, turns a Hugging Face dataset into a
handful of artifacts, and stops. The **serving app** reads those artifacts and
answers HTTP. Nothing is computed per-request that could have been computed once,
with one exception noted below, and the server holds no mutable state: every
response is a pure function of `DATASET_REVISION` and the files on disk.

```
 hanbangwu/alphauniverse-cosmos (Hugging Face)
              │
              │  generate_embeddings   GPU, ~3h
              ▼
   encoded · codebook · tokens  ──────────────┐
              │                               │
              │  generate_index   CPU          │  generate_projections   GPU
              ▼                               ▼
      encoded_index.faiss            parametric_umap · mean_points · full_points
              │                               │
              └───────────┬───────────────────┘
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

| Role              | Shape                                          | Who reads it                |
| ----------------- | ---------------------------------------------- | --------------------------- |
| `encoded`         | one row per galaxy, embeddings per survey      | index build, projections, download |
| `codebook`        | same, raw codebook vectors                     | nothing yet                 |
| `tokens`          | same, token ids                                | `/galaxies/{g}/tokens`, `/coverage` |
| `encoded_index`   | faiss IVF over anchor image patches            | `/similarity`               |
| `mean_points`     | one 2-d point per galaxy                       | `/meta`, the projection view |
| `full_points`     | one 2-d point per embedding                    | the projection view         |
| `parametric_umap` | the trained projector's weights                | nothing at serve time       |

`docs/pipeline.md` has the schemas.

## The serving app

Six endpoints, all `GET`, all cacheable by revision.

| Endpoint                     | Returns                  | Cost per request                        |
| ---------------------------- | ------------------------ | --------------------------------------- |
| `/meta`                      | JSON                     | cached after the first call             |
| `/artifacts/{role}`          | the file                 | disk read, streamed; up to tens of GB   |
| `/galaxies/{g}/image.png`    | PNG                      | decode + crop + re-encode, every time   |
| `/galaxies/{g}/tokens`       | raw `uint32`             | one filtered parquet read               |
| `/galaxies/{g}/coverage`     | JSON                     | one filtered parquet read               |
| `/similarity`                | Arrow IPC                | one ANN search + candidate rescoring    |

The one thing recomputed per request that need not be is the cutout: `/image.png`
decodes the galaxy's image out of the Hugging Face dataset, centre-crops it and
re-encodes a PNG on every call, with nothing cached and no cache headers on the
response.

Bulk payloads avoid JSON deliberately. Token maps are raw little-endian `uint32`,
similarity results are a single Arrow record batch, and the point sets stay as
parquet so the browser can query them with DuckDB instead of the server
paginating them.

`Meta` returns artifact *roles*, not URLs — `embeddings`, `mean_points`,
`full_points` are names the client hands back to `/artifacts/{role}`. That keeps
the server in charge of which artifact backs which view.

## Similarity search

The interesting path. A query is some patches of one galaxy; the answer is a
ranked list of galaxies, each with a full per-patch score map.

1. Reconstruct the query patches from the index and average them into one
   direction, L2-normalised.
2. One ANN search for the `PROBE` (2048) nearest patches anywhere in the survey.
3. Fold those patches to their galaxies, ordered by best-ranked patch, and keep
   the first `matches`.
4. Reconstruct *every* patch of every candidate and score it against the query
   direction. Rank candidates by their best patch.

So the candidate set is approximate and the ranking within it is exact. Two
consequences worth knowing:

- The result can be **shorter than `matches`**. If 2048 patches happen to belong
  to few galaxies, that is all you get, and nothing in the response says so.
- Step 4 does far more work than step 2 — about 90% of query time. See
  `docs/performance.md`.

Vector ids carry position: `id = galaxy * N_PATCHES + patch`. That is what makes
steps 1 and 4 possible without a side table, and it holds only because the index
build adds every galaxy's 576 anchor patches in row order.

## The frontend

SvelteKit, Svelte 5 runes, one page in three resizable panes.

```
+page.svelte
├── LeftPanel        point set toggle, morphology filter, download
├── ProjectionView   embedding-atlas over a DuckDB-WASM table
└── RightPanel       selected galaxy: cutout, tokens, coverage
    └── PatchSimilarity   dialog: query patches, ranked matches
```

State lives in plain classes under `src/lib/state/`, held in a `runed` context
and reached with `getApp()`, `getView()`, `getSearch()`, `getMask()`,
`getFilters()`. `Field<T>` and its subclasses wrap a `$state` value with
normalisation — clamping numbers, sorting and deduping index lists — so the
components never validate anything themselves. Numeric bounds are not typed by
hand: `state/schema.ts` reads them out of the generated zod schema, so the
slider ranges come from the API contract.

Two data paths, deliberately separate:

- **Row-level data** (`meta`, tokens, coverage, similarity) goes through the
  generated client into TanStack Query. Everything keyed by revision is cached
  forever; similarity has a 5-minute stale time.
- **The point sets** never pass through the app. DuckDB-WASM fetches the parquet
  artifact directly over range requests, and Mosaic pushes the morphology filter
  into SQL so filtering the projection never round-trips to the server.

Patch grids render to `<canvas>` at native resolution — one `ImageData` of
24×24 pixels scaled up with `image-rendering: pixelated` — plus a second canvas
for selection and hover outlines. `IsInViewport` gates the redraw, which matters
because a full match list is ~32 rows of three grids each.

## Deployment

`main` pushes deploy to Modal via `.github/workflows/ci-cd.yml`, gated on tests,
lint and an OpenAPI drift check. The frontend deploys separately on Vercel.

The serving function is pinned to `max_containers=1` with `scaledown_window` of
5 minutes and 16 concurrent inputs, so the whole service is one process that
disappears after five idle minutes and reloads a ~16 GB index on the next
request. That single fact dominates first-visit latency; `docs/performance.md`
covers it.
