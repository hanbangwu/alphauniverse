# Performance

Where the time goes, what has been measured, and what has not.

Everything labelled **measured** comes from `scripts/benchmark.py` against a
32-galaxy fixture on a development machine. Everything labelled **projected** is
that fixture scaled linearly to the production 17,369 galaxies, which is sound
for sizes and misleading for anything involving index geometry — see
[Reading fixture numbers](#reading-fixture-numbers). Everything labelled
**unmeasured** is read off the code and needs a real number before it is trusted.

## The cost model

A cold visit pays, in order: the serving container starting, a ~16 GB index
loading, SSR blocking on `/meta`, a point-set parquet downloading, DuckDB-WASM
booting, and then the first render. A warm interaction pays one HTTP round trip
and a few tens of milliseconds of server compute.

The gap between those two is the whole problem. Server compute, per request, is
already small.

## Measured: query latency

p50, 32-galaxy fixture, 30 runs.

| Query shape                | p50     | p95     |
| -------------------------- | ------- | ------- |
| 1 patch, 8 matches         | 9.9 ms  | 15.5 ms |
| 4 patches, 8 matches       | 9.1 ms  | 15.6 ms |
| 16 patches, 8 matches      | 10.2 ms | 16.8 ms |
| 1 patch, 32 matches        | 35.2 ms | 46.5 ms |
| 16 patches, 32 matches     | 35.1 ms | 46.9 ms |
| any, 128 matches           | 32.1 ms | 39.9 ms |

**Cost is set by `matches`, not by how many patches are queried.** The patch
count changes one averaging step over a handful of vectors; `matches` changes how
many galaxies get fully rescored. (128 matches ties with 32 only because the
fixture has just 32 galaxies to return.)

Splitting one query into stages says why:

| Stage                            | p50      | Share  |
| -------------------------------- | -------- | ------ |
| Reconstruct query direction      | 0.54 ms  | 1.7 %  |
| ANN search, `PROBE=2048`         | 0.89 ms  | 2.8 %  |
| Fold patches to galaxies         | 0.18 ms  | 0.6 %  |
| **Reconstruct candidate maps**   | **28.9 ms** | **90.2 %** |
| Score matmul                     | 1.55 ms  | 4.8 %  |

Step 4 reconstructs 18,432 vectors at **1.57 µs each**. That is roughly seven
times what decoding 768 fp16 values and dotting them should cost, because
`reconstruct_batch` walks the IVF direct map one vector at a time: a list lookup
and a per-vector decode call rather than a contiguous read.

The approximate search — the part that sounds expensive — is 3% of the query.

## Measured: everything else

| Endpoint                 | p50    | Note                                  |
| ------------------------ | ------ | ------------------------------------- |
| `/meta`                  | 1.1 ms | cached in-process after the first call |
| `/galaxies/{g}/tokens`   | 2.4 ms | filtered parquet read                 |
| `/galaxies/{g}/coverage` | 2.8 ms | filtered parquet read                 |

Not measured here: `/galaxies/{g}/image.png`, which needs the real dataset.

## Projected: artifact sizes

| Artifact          | 32-galaxy fixture | Projected at 17,369 |
| ----------------- | ----------------- | ------------------- |
| `encoded`         | 48.4 MB           | **26.3 GB**         |
| `encoded_index`   | 29.9 MB           | **16.2 GB**         |
| `full_points`     | 0.21 MB           | ~0.12 GB *(low)*    |
| `tokens`          | 0.03 MB           | 0.02 GB             |
| `mean_points`     | 1.6 KB            | ~1 MB               |

The index projection matches an independent calculation: 17,369 × 576 patches
= 10,004,544 vectors × 768 dims × 2 bytes for `SQfp16` = 15.4 GB, plus ids and
the direct map. Call it **15–16 GB, resident in RAM, on every container**.

`full_points` is projected low: the fixture writes anchor rows only, while
production also carries HSC, DESI and SDSS points. Real size is likely 150–250
MB. It needs measuring against the real artifact.

## Ranked opportunities

Roughly by user-visible latency saved per unit of work, highest first.

### 1. Cold start — unmeasured, almost certainly dominant

The serving function runs `max_containers=1` with a 5-minute `scaledown_window`,
and `faiss.read_index` pulls the whole ~16 GB index into memory with no mmap.
Any visitor arriving more than five minutes after the last one waits for a 16 GB
read from a Modal network volume before a single byte of `/meta` comes back —
and `+layout.server.ts` blocks SSR on `/meta`, so nothing renders until it does.

Levers, in increasing cost: `faiss.IO_FLAG_MMAP` so pages load lazily; a smaller
index (`SQ8` halves it, PQ far more, both at some recall cost); `min_containers=1`
to keep one warm. The first is nearly free and should be tried first. **Measure
the real cold start before choosing.**

### 2. No cache headers anywhere — unmeasured, nearly free

Every response is immutable for a given `DATASET_REVISION`, and not one of them
sets `Cache-Control`. Every repeat visit re-fetches everything. `FileResponse`
does emit `etag`/`last-modified`, so artifacts at least revalidate; `/meta`,
`/tokens`, `/coverage`, `/similarity` and `/image.png` do not even do that.

Putting the revision in the path or a query parameter makes the whole surface
safely `immutable`, at which point repeat visits cost nothing.

### 3. Candidate reconstruction — measured, 90% of query compute

Keep the anchor patches as one contiguous memory-mapped fp16 array beside the
index and slice `galaxy * 576 … (galaxy + 1) * 576` out of it, instead of asking
faiss to reconstruct vector by vector. A 33-galaxy result is 29 MB of sequential
reads and one BLAS matmul. This should take a 32 ms query to well under 10 ms.

Worth doing, but note it optimises 32 ms — item 1 is worth seconds.

### 4. `/artifacts/{role}` through the app container — unmeasured

`full_points` (~150–250 MB) and `encoded` (~26 GB, wired to a "Download
embeddings" button in the UI) stream through the single serving container. One
download can starve every interactive request behind it. Object storage or a CDN
in front is the structural fix; at minimum, `encoded` should not be a one-click
button on a `max_containers=1` service.

### 5. Cutouts re-encoded per request — unmeasured

`/galaxies/{g}/image.png` decodes from the Hugging Face dataset, centre-crops and
re-encodes a PNG every time, uncached. Opening the similarity dialog fires ~32 of
these at once against a 16-input container. Precomputing 96×96 PNGs into an
artifact — 17,369 × 96 × 96 × 3 ≈ 480 MB raw, far less as PNG — turns this into a
range read, and makes it CDN-able.

### 6. Frontend — unmeasured

- **SSR blocks on `/meta`.** A cold backend stalls the document, not just a
  widget. Moving it client-side would let the shell paint immediately.
- **`full_points` is materialised**, not scanned: `loadParquet` runs
  `CREATE TABLE … AS SELECT`, so the whole table sits in WASM memory.
- **The match list is not virtualised.** Up to 128 rows × 3 patch grids, each
  with two canvases. `IsInViewport` gates redraws, which is why this is tolerable
  today, but it caps how far `matches` can usefully go.
- **No URL state.** The selected galaxy, patches and filters live only in memory,
  so nothing is shareable or survives a reload.

## Reading fixture numbers

The fixture is small, and small changes the index's geometry. Two ratios matter
and both are wrong at fixture scale:

| | Fixture (32 galaxies) | Production |
| --- | --- | --- |
| `nlist` | 472 | 16,384 |
| `nprobe / nlist` | 13.6 % | **0.39 %** |
| `PROBE` as a share of the corpus | 11 % | **0.02 %** |

The fixture probes 35× more of its index and its candidate pool covers 11% of
all patches. Latency ordering and the stage split carry over; **absolute recall
does not**.

Measured recall against brute force over every patch, 16 queries, 31 requested:

| `nprobe` | Recall | Galaxies returned |
| -------- | ------ | ----------------- |
| 1        | 0.68   | 21.1 / 31         |
| 4        | 1.00   | 30.9 / 31         |
| 16 – 472 | 1.00   | 31.0 / 31         |

What this does establish: the metric is defined and wired up, and the
"shorter than `matches`" behaviour is real and silent — at `nprobe=1` a third of
the requested galaxies simply never come back, with nothing in the response
saying so. **Production recall is unknown** and needs measuring against the real
index before any change to `NPROBE`, `PROBE`, `NLIST` or the quantiser.

## Scaling ceilings

Current design targets COSMOS scale. Where it stops:

| Ceiling | Now | Breaks at |
| ------- | --- | --------- |
| Index in RAM | 15–16 GB | ~4× — a 64 GB container. 100× needs PQ or sharding. |
| `full_points` in the browser | ~150–250 MB | ~10× — DuckDB-WASM has a few GB to work with. |
| Exact-search reference | whole corpus in RAM | already fixture-only; `anchor_patches()` cannot run at production scale |
| Serving capacity | one container | any concurrency at all; `max_containers=1` is a hard cap |
| `GalaxyIndex` | hardcoded 17,369 | any revision with a different galaxy count |

None of these need solving now. All of them should be checked before a change
assumes they are not there.

## Running the benchmark

```sh
uv run python -m scripts.benchmark --galaxies 32 --out bench.json
```

Results are tagged with the commit and the fixture shape. Numbers only compare
between runs with the same `--galaxies` and `--seed`. See `docs/testing.md`.
