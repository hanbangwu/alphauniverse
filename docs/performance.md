# Performance

Where the time goes, what has been measured, and what has not.

Numbers come from two places, and the difference matters.

**Production** figures were taken against the deployed API at
`surp-2026--alphauniverse-fastapi-app.modal.run` over the public internet, so
they include network round trip — about 240 ms from the measuring host, which is
subtracted where a figure is labelled *compute*. **Fixture** figures come from
`scripts/benchmark.py` against a 32-galaxy tree on a development machine; they
are reproducible and CI-runnable but do not carry over directly, for the reasons
in [Reading fixture numbers](#reading-fixture-numbers).

Anything labelled **unmeasured** is read off the code and needs a real number
before it is trusted.

## The cost model

Measured against production, the path a first visitor to an idle site walks:

| Step | Cost |
| ---- | ---- |
| Container start + 15.5 GB index load, blocking `/meta` | **47.0 s** |
| First cutout, which also opens the Hugging Face dataset | **11.3 s** |
| Everything warm after that | 0.5–0.7 s per request |

**A cold visit is ~47 seconds of blank page**, because `+layout.server.ts` awaits
`/meta` during SSR and nothing renders until it returns. Opening patch similarity
then costs ~100 ms of search plus 11 s of thumbnails.

Warm, the service is fine. Two things dominate, and neither is the search
algorithm:

1. **Cold start** — 15.5 GB read into RAM, one container, discarded after five
   idle minutes.
2. **Work repeated per request that could be done once** — cutouts above all,
   and everything a cache header would have avoided.

The approximate nearest-neighbour search — the part that sounds expensive — is
3% of a query.

## Production: query latency

`/similarity` against the real index, best of 3 per shape, with the 240 ms
network baseline subtracted.

| Query shape             | Compute  |
| ----------------------- | -------- |
| 1 patch, 8 matches      | ~25 ms   |
| 16 patches, 8 matches   | ~30 ms   |
| 1 patch, 32 matches     | ~105 ms  |
| 16 patches, 32 matches  | ~95 ms   |
| 1 patch, 128 matches    | ~328 ms  |
| 16 patches, 128 matches | ~321 ms  |

**Cost is set by `matches` and is independent of how many patches are queried.**
Each 4× in `matches` costs 3–4× the time (25 → 105 → 328 ms). The patch count
changes one averaging step over a handful of vectors; `matches` changes how many
galaxies get fully rescored, at 576 vector reconstructions each.

The default of 32 matches costs ~100 ms of server compute; the UI's maximum of
128 costs ~325 ms. That works out to **4.4–5.3 µs per reconstructed vector** —
three times the fixture's 1.57 µs, which is what an index 543× larger does to
cache locality on the direct-map lookups.

## Fixture: query latency

p50, 32-galaxy fixture, 30 runs. Useful because the stages can be timed
separately; the absolute numbers are optimistic.

| Query shape                | p50     | p95     |
| -------------------------- | ------- | ------- |
| 1 patch, 8 matches         | 9.9 ms  | 15.5 ms |
| 4 patches, 8 matches       | 9.1 ms  | 15.6 ms |
| 16 patches, 8 matches      | 10.2 ms | 16.8 ms |
| 1 patch, 32 matches        | 35.2 ms | 46.5 ms |
| 16 patches, 32 matches     | 35.1 ms | 46.9 ms |
| any, 128 matches           | 32.1 ms | 39.9 ms |

(128 matches ties with 32 here only because the fixture has just 32 galaxies to
return. Production, with 17,369 to draw from, separates them cleanly.)

Splitting one query into stages says why cost tracks `matches`:

| Stage                            | p50      | Share  |
| -------------------------------- | -------- | ------ |
| Reconstruct query direction      | 0.54 ms  | 1.7 %  |
| ANN search, `PROBE=2048`         | 0.89 ms  | 2.8 %  |
| Fold patches to galaxies         | 0.18 ms  | 0.6 %  |
| **Reconstruct candidate maps**   | **28.9 ms** | **90.2 %** |
| Score matmul                     | 1.55 ms  | 4.8 %  |

Step 4 reconstructs 18,432 vectors at **1.57 µs each**, because
`reconstruct_batch` walks the IVF direct map one vector at a time: a list lookup
and a per-vector decode call rather than a contiguous read.

The approximate search — the part that sounds expensive — is 3% of the query.

## Production: cutouts

This is the worst number in the system.

| Pattern                              | Result                        |
| ------------------------------------ | ----------------------------- |
| First request after container start  | **11.3 s**                    |
| Subsequent, sequential               | 0.43–0.66 s each              |
| **32 in parallel**                   | **11.4 s wall, 7.8 s median** |

Opening the patch-similarity dialog renders up to 32 match rows, each with an
`<img>`. Those 32 requests take **11 seconds** before the last thumbnail appears.

The 32-way burst takes the same wall clock as running the 32 sequentially would,
so `@modal.concurrent(max_inputs=16)` is buying nothing here — the work is
CPU-bound PIL decode, crop and re-encode inside one Python process, and the
concurrency setting just queues it. Throughput is ~2.8 requests/second either
way.

The 11.3 s first request is separate and reproducible — it measured 11.32 s and
11.31 s on two container starts. `dataset()` is lazily cached, so the first
cutout after a container start also pays for opening the Hugging Face dataset.
`lifespan` preloads the index and the token store but not this, so the cost
lands on a user instead of on startup.

## Fixture: metadata endpoints

In-process, no network.

| Endpoint                 | p50    | Note                                   |
| ------------------------ | ------ | -------------------------------------- |
| `/meta`                  | 1.1 ms | cached in-process after the first call |
| `/galaxies/{g}/tokens`   | 2.4 ms | filtered parquet read                  |
| `/galaxies/{g}/coverage` | 2.8 ms | filtered parquet read                  |

Against production these sit at the ~240 ms network floor, so their compute is
negligible either way.

## Production: artifact sizes

Measured with `HEAD /artifacts/{role}`.

| Artifact          | Size        | Notes                                     |
| ----------------- | ----------- | ----------------------------------------- |
| `encoded`         | **22.97 GB** | wired to a "Download embeddings" button   |
| `encoded_index`   | **15.50 GB** | read into RAM on every container start    |
| `codebook`        | **2.70 GB**  | same shape as `encoded`, 8.5× smaller     |
| `full_points`     | **180 MB**   | downloaded into the browser on "Full"     |
| `tokens`          | **7.0 MB**   |                                           |
| `parametric_umap` | **923 KB**   | not read at serve time                    |
| `mean_points`     | **259 KB**   | downloaded on first paint                 |

The index size matches an arithmetic check: 17,369 × 576 patches = 10,004,544
vectors × 768 dims × 2 bytes for `SQfp16` = 15.37 GB, plus ids and the direct
map. **15.5 GB, resident in RAM, on every container.**

`codebook` being 8.5× smaller than `encoded` despite an identical schema is worth
noting: codebook vectors are drawn from a finite codebook, so the same 768-d rows
repeat and zstd collapses them. Contextualised outputs are all distinct and do
not compress. Any future work that can use codebook embeddings instead of encoded
ones gets an order of magnitude on storage for free.

**No endpoint sets `Cache-Control`.** Artifacts carry an `etag`, so they at least
revalidate; `/meta`, `/tokens`, `/coverage`, `/similarity` and `/image.png` carry
neither.

## Ranked opportunities

By measured user-visible latency saved, highest first.

### 1. Cold start — 47 s of blank page

Measured: 400 s idle, then `GET /meta` took **46.97 s**, against 0.47 s warm.
That is `faiss.read_index` pulling the whole 15.5 GB index into memory with no
mmap — roughly 330 MB/s off the Modal volume — while `max_containers=1` means
there is no warm sibling to answer instead. Because SSR awaits `/meta`, the
visitor sees nothing at all for those 47 seconds, not even a loading state.

Levers, cheapest first:

- **`faiss.IO_FLAG_MMAP`** so pages fault in lazily. A search at `nprobe=64`
  touches ~0.4% of the lists, so time-to-first-response should drop by a large
  factor even though steady-state stays the same.
- **Preload `dataset()` in `lifespan`**, which already warms the index and token
  store. Moves the 11.3 s dataset open off the first user.
- **Don't block SSR on `/meta`** — fetch it client-side so the shell paints
  immediately and the wait becomes a spinner instead of nothing.
- **A smaller index** — `SQ8` halves it, PQ far more, both at some recall cost
  that `scripts/benchmark.py` can quantify first.
- **`min_containers=1`** removes it entirely, trading money for latency.

The first three are cheap and compose. Do them before considering the last.

### 2. Cutouts re-encoded per request — 11 s to open the match list

`/galaxies/{g}/image.png` reads the galaxy's image out of the Hugging Face
dataset, centre-crops it and re-encodes a PNG on every call. Nothing is cached,
on the server or in the response. At ~350 ms of CPU each and no useful
concurrency, the 32 thumbnails in a match list take 11 seconds.

Precompute them. Every cutout is a pure function of the revision: 17,369 × ~10.8
KB is about **190 MB** as a single artifact, roughly 100 minutes of one-off CPU.
Serving then becomes a range read with no decode at all, and it becomes
CDN-cacheable, which the current endpoint can never be.

This is the largest measured win available and it needs no algorithmic work.

### 3. No cache headers anywhere — nearly free

Every response is immutable for a given `DATASET_REVISION`, and not one sets
`Cache-Control`. Artifacts carry an `etag` so they revalidate; `/meta`,
`/tokens`, `/coverage`, `/similarity` and `/image.png` carry nothing, so every
repeat visit re-computes and re-transfers everything.

Putting the revision in the path or a query parameter makes the whole surface
safely `immutable`, after which repeat visits cost nothing and a CDN can absorb
the traffic. This is the cheapest change in this document.

### 4. Candidate reconstruction — 100 ms at the default, 325 ms at the maximum

90% of a query is `reconstruct_batch` walking the IVF direct map one vector at a
time, at 4.4–5.3 µs per vector in production. The ANN search is 3%.

Keep the anchor patches as one contiguous memory-mapped fp16 array beside the
index and slice `galaxy * 576 … (galaxy + 1) * 576` out of it. A 33-galaxy result
is then 29 MB of sequential reads and one BLAS matmul, which should land under
15 ms — call it **7–10×** on this endpoint, taking the 128-match case from
~325 ms to tens of milliseconds.

That array is 15.4 GB, the same data the index already holds, so this trades
memory for speed unless the index itself is compressed at the same time. Worth
planning the two together.

### 5. Bulk artifacts through the app container

`encoded` is **22.97 GB** and is wired to a "Download embeddings" button in the
left panel. `full_points` is **180 MB** and downloads whenever a user switches to
the full point set. Both stream through the single serving container, where one
download can starve every interactive request behind it.

Object storage or a CDN in front is the structural fix. At minimum, a 23 GB
download should not be a one-click button on a `max_containers=1` service.

### 6. Frontend

- **SSR blocks on `/meta`.** A cold backend stalls the document itself, not just
  a widget. Moving the call client-side would let the shell paint immediately
  and show a loading state instead of nothing.
- **`full_points` is materialised**, not scanned: `loadParquet` runs
  `CREATE TABLE … AS SELECT`, so all 180 MB is decoded into WASM memory.
- **The match list is not virtualised.** Up to 128 rows × 3 patch grids, each
  with two canvases. `IsInViewport` gates redraws, which is the only reason this
  is tolerable, but it caps how far `matches` can usefully go — and at 128 the
  server is already spending 325 ms plus 128 cutout requests.
- **No URL state.** The selected galaxy, patches and filters live only in
  memory, so nothing is shareable or survives a reload.

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
| Index in RAM | 15.5 GB | ~4× — the container's 64 GB limit. 100× needs PQ or sharding. |
| Cold start | 47 s | ~2× before it exceeds common proxy and browser timeouts |
| `full_points` in the browser | 180 MB | ~10× — DuckDB-WASM has a few GB to work with. |
| Cutout precompute | 190 MB, ~100 min CPU | linear; fine to ~100×, then needs tiling |
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
