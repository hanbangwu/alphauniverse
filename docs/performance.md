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

Measured against production, what a first visitor to an idle site waits for:

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

The approximate nearest-neighbour search is 3% of a query.

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

The approximate search is 3% of the query.

## Production: cutouts

This is the largest measured latency in the system.

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
ones costs an order of magnitude less to store.

**No endpoint sets `Cache-Control`.** Artifacts carry an `etag`, so they at least
revalidate; `/meta`, `/tokens`, `/coverage`, `/similarity` and `/image.png` carry
neither.

## Candidate work

Nothing in this section is done. Each entry is triaged on three axes, because
they answer different questions and a change can score well on one and badly on
another:

- **Impact** — user-visible latency or compute saved, read off the measurements
  above rather than off intuition.
- **Blast radius** — how much code, and how much of the search method, a change
  disturbs. These come apart: a one-line change that alters how every query
  touches memory has a small diff and a large blast radius.
- **Kind** — *implementation* pays off directly, *quality* removes a defect or
  an obstacle without saving time, *groundwork* only makes a later change
  possible or provable.

Entries are named rather than numbered. The table is ordered by measured impact
and gets re-ordered when a measurement changes, so a number would be a
cross-reference that goes stale; the name is stable.

| Entry | Impact | Blast radius | Kind |
| ----- | ------ | ------------ | ---- |
| `index-mmap` | 47 s cold start, if faiss supports it here | Small diff, large behaviour | Implementation |
| `cutout-artifact` | 11 s off a match list | Medium-large | Implementation |
| `ssr-unblock` | Turns 47 s of blank page into 47 s of spinner | Medium, frontend | Quality |
| `dataset-preload` | 11.3 s off the first user per container | Trivial | Implementation |
| `patch-array` | 7–10× on `/similarity` | Large | Implementation |
| `cache-headers` | Whole surface on repeat visits; magnitude unmeasured | Small | Implementation |
| `index-compression` | Cold start, and the RAM ceiling | Large, methodological | Implementation |
| `tokens-bulk` | ~240 ms per newly selected galaxy | Medium | Implementation |
| `bulk-offload` | None steady-state, large under load | Infrastructure | Quality |
| `full-points-view` | Browser memory and time-to-full-view; unmeasured | Small-medium | Implementation |
| `match-list-virtualise` | Low | Medium, frontend | Quality |
| `min-containers` | Removes cold start outright | None | Operational, costs money |
| `production-recall` | None directly | Small | Groundwork |
| `search-stages` | None | Small | Groundwork |
| `galaxy-index-derived` | None | Small | Quality |
| `recall-reported` | None | Small | Quality |
| `encode-coverage` | None | Small | Groundwork |

### Sequencing

Impact order is not the order to work in, because of four couplings.

**`dataset-preload` must not block startup.** Preloading the Hugging Face
dataset moves 11.3 s off the first user and onto startup. Startup is currently
47 s of blank page, so a blocking preload makes the blank page 58 s. Either warm
it in the background after `lifespan` yields, or block only once `index-mmap`
has made the index read cheap.

**`index-mmap` needs a spike before it counts as small.** `write_index` stores
an IVF index's inverted lists in the in-memory array form, and faiss's
`IO_FLAG_MMAP` may require the on-disk form instead. Even if it loads,
`make_direct_map()` plus page faults during `reconstruct_batch` could trade 46 s
of startup for unpredictable per-query latency. The change is one flag; the
question it raises is whether every query gets slower, and only a measurement
answers it.

**`patch-array` and `index-compression` are one design, not two changes.**
`patch-array` adds a 15.4 GB contiguous fp16 array holding the same vectors the
index already holds, against a 64 GB container. `index-compression` shrinks the
index at some cost in recall. Done together they separate two jobs the index is
currently doing at once: a heavily compressed index generates candidates, and
the flat array scores them exactly. Done separately, the first doubles memory
and the second loses accuracy for nothing.

**`production-recall` gates both of them, and `search-stages` gates the
benchmark that would prove them.** Production recall is unknown, for the reasons
in [Reading fixture numbers](#reading-fixture-numbers), so there is no baseline
to show a quantiser change did not silently degrade results. And
`measure_phases` in `scripts/benchmark.py` re-implements `search()` step by step
to time its stages, which means the benchmark and the code can drift apart
precisely when they most need to agree.

That gives a working order. `cache-headers`, `cutout-artifact` and a
non-blocking `dataset-preload` are independent and provable now. `search-stages`
is the refactor that makes everything after it measurable against the code that
actually runs. Then the `index-mmap` spike and `production-recall`, which
between them unlock the `patch-array` + `index-compression` design.

### Cold start — 47 s of blank page

Measured: 400 s idle, then `GET /meta` took **46.97 s**, against 0.47 s warm.
That is `faiss.read_index` pulling the whole 15.5 GB index into memory with no
mmap — roughly 330 MB/s off the Modal volume — while `max_containers=1` means
there is no warm sibling to answer instead. Because SSR awaits `/meta`, the
visitor sees nothing at all for those 47 seconds, not even a loading state.

With `scaledown_window=5*60` this is not a tail case. Any visitor arriving more
than five minutes after the last one pays it in full, so on a low-traffic site
it is the common experience rather than the rare one.

Options, cheapest first:

- **`index-mmap`** — `faiss.IO_FLAG_MMAP` so pages fault in lazily. A search at
  `nprobe=64` touches ~0.4% of the lists, so time-to-first-response should drop
  by a large factor even though steady-state stays the same.
- **`dataset-preload`** — warm `dataset()` in `lifespan`, which already warms
  the index and token store. Moves the 11.3 s dataset open off the first user.
- **`ssr-unblock`** — fetch `/meta` client-side so the shell paints immediately
  and the wait becomes a spinner instead of nothing.
- **`index-compression`** — `SQ8` halves the index, PQ far more, both at some
  recall cost that `scripts/benchmark.py` can quantify first.
- **`min-containers`** — `min_containers=1` removes cold start entirely,
  trading money for latency.

The first three are cheap and compose. Do them before considering the last two.

### Cutouts re-encoded per request — 11 s to open the match list

`/galaxies/{g}/image.png` reads the galaxy's image out of the Hugging Face
dataset, centre-crops it and re-encodes a PNG on every call. Nothing is cached,
on the server or in the response. At ~350 ms of CPU each and no useful
concurrency, the 32 thumbnails in a match list take 11 seconds.

`cutout-artifact` precomputes them. Every cutout is a pure function of the
revision: 17,369 × ~10.8 KB is about **190 MB** as a single artifact, roughly
100 minutes of one-off CPU. Serving then becomes a range read with no decode at
all, and it becomes CDN-cacheable, which the current endpoint can never be.

This is the largest measured win available and it needs no algorithmic work.

### No cache headers anywhere — nearly free

Every response is immutable for a given `DATASET_REVISION`, and not one sets
`Cache-Control`. Artifacts carry an `etag` so they revalidate; `/meta`,
`/tokens`, `/coverage`, `/similarity` and `/image.png` carry nothing, so every
repeat visit re-computes and re-transfers everything.

The one design question in `cache-headers` is what makes the immutability safe
to advertise. A long `max-age` with `immutable` is correct only while the
revision does not change; a browser that cached under the old revision would
keep serving it. Putting the revision in the path or a query parameter makes the
whole surface safely `immutable`, after which repeat visits cost nothing and a
CDN can absorb the traffic.

### Candidate reconstruction — 100 ms at the default, 325 ms at the maximum

90% of a query is `reconstruct_batch` walking the IVF direct map one vector at a
time, at 4.4–5.3 µs per vector in production. The ANN search is 3%.

`patch-array` keeps the anchor patches as one contiguous memory-mapped fp16
array beside the index and slices `galaxy * 576 … (galaxy + 1) * 576` out of it.
A 33-galaxy result is then 29 MB of sequential reads and one BLAS matmul, which
should land under 15 ms — call it **7–10×** on this endpoint, taking the
128-match case from ~325 ms to tens of milliseconds.

That array is 15.4 GB, the same data the index already holds, so it only makes
sense alongside `index-compression`. See [Sequencing](#sequencing).

### Bulk artifacts through the app container

`encoded` is **22.97 GB** and is wired to a "Download embeddings" button in the
left panel. `full_points` is **180 MB** and downloads whenever a user switches to
the full point set. Both stream through the single serving container, where one
download can starve every interactive request behind it.

`bulk-offload` puts object storage or a CDN in front. At minimum, a 23 GB
download should not be a one-click button on a `max_containers=1` service.

### Frontend

- **`ssr-unblock`** — `+layout.server.ts` awaits `/meta`, so a cold backend
  stalls the document itself rather than a widget. The cost is that every
  consumer of `data.meta` must then handle its absence.
- **`full-points-view`** — `loadParquet` runs `CREATE TABLE … AS SELECT`, so all
  180 MB is decoded into WASM memory. A view over the parquet would read ranges
  on demand instead, but whether that is faster depends on how many queries
  follow and how well DuckDB-WASM caches those ranges. Unmeasured either way.
- **`match-list-virtualise`** — up to 128 rows × 3 patch grids, each with two
  canvases. `IsInViewport` gates redraws, which is the only reason this is
  tolerable, but it caps how far `matches` can usefully go.
- **No URL state.** The selected galaxy, patches and filters live only in
  memory, so nothing is shareable or survives a reload.

### Smaller items

- **`tokens-bulk`** — `/galaxies/{g}/tokens` returns 2.3 KB and costs a ~240 ms
  round trip, paid once per galaxy the user selects. The whole `tokens` artifact
  is 7.0 MB, so shipping it once and querying in-browser could remove every one
  of those round trips. Whether it pays depends on the anchor column's share of
  that 7 MB, which is unmeasured, and on how many galaxies a session touches.
- **`galaxy-index-derived`** — `GalaxyIndex` hardcodes 17,369 instead of reading
  the artifacts, so any revision with a different galaxy count silently accepts
  or rejects the wrong indices. Tracked by the strict xfail in
  `tests/test_api.py`.
- **`recall-reported`** — `search()` can return fewer than `matches` galaxies
  when the candidate patches do not cover enough of them, and the response says
  nothing about it. At `nprobe=1` in the fixture a third of the requested
  galaxies silently never came back.
- **`encode-coverage`** — `app/encode.py` has no automated coverage at all, so
  an import error or a schema drift there is caught only by a full build. It
  imports torch at module level, so a test needs the stubbed-import approach
  rather than a real import.

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
the requested galaxies never come back, with nothing in the response
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
