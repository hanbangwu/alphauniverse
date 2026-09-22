# Performance

Numbers come from two places:

- **Production** figures were taken against the deployed API over the public internet, so they include network round trip: about 240 ms from the measuring host, which is subtracted where a figure is labelled _compute_.
- **Fixture** figures come from `scripts/benchmark.py` against a 32-galaxy tree on a development machine; they are reproducible and CI-runnable but do not carry over directly, for the reasons in [Reading fixture numbers](#reading-fixture-numbers).

Anything labelled **unmeasured** is read off the code and needs a real number before it is trusted.

## The cost model

Measured against production, what a first visitor to an idle site waits for:

| Step                                                   | Cost                  |
| ------------------------------------------------------ | --------------------- |
| Container start + 15.5 GB index load, blocking `/meta` | **47.0 s**            |
| Everything warm after that                             | 0.5–0.7 s per request |

**A cold visit is ~47 seconds of blank page**, because `+layout.server.ts` awaits `/meta` during SSR and nothing renders until it returns.

Cold start dominates, and it is not the search algorithm: the approximate nearest-neighbour search is 3% of a query.

## Production: query latency

`/similarity` against the real index, best of 3 per shape, with the 240 ms network baseline subtracted. These predate the spectral tokens joining the index: a query now also reconstructs 272 spans for each candidate with a spectrum, unmeasured.

| Query shape             | Compute |
| ----------------------- | ------- |
| 1 patch, 8 matches      | ~25 ms  |
| 16 patches, 8 matches   | ~30 ms  |
| 1 patch, 32 matches     | ~105 ms |
| 16 patches, 32 matches  | ~95 ms  |
| 1 patch, 128 matches    | ~328 ms |
| 16 patches, 128 matches | ~321 ms |

**Cost is set by `matches` and is independent of how many patches are queried.** Each 4× in `matches` costs 3–4× the time (25 → 105 → 328 ms). The patch count changes one averaging step over a handful of vectors; `matches` changes how many galaxies get fully rescored, at 576 vector reconstructions each.

The default of 32 matches costs ~100 ms of server compute; the UI's maximum of 128 costs ~325 ms. That works out to **4.4–5.3 µs per reconstructed vector**, against about 1 µs for the fixture. The gap is what an index 543× larger does to cache locality on the direct-map lookups, though the fixture figure moves with the host, so the multiple is approximate.

## Fixture: query latency

p50, 32-galaxy fixture, 30 runs, seed 0, on a four-core host with faiss using four threads. Useful because the stages can be timed separately; the absolute numbers are optimistic. Every table in this section comes from that one run, and a run's `environment` block records the thread configuration, because the next section is why that matters.

| Query shape            | p50      | p95      |
| ---------------------- | -------- | -------- |
| 1 patch, 8 matches     | 8.0 ms   | 10.2 ms  |
| 4 patches, 8 matches   | 8.0 ms   | 16.1 ms  |
| 16 patches, 8 matches  | 8.0 ms   | 14.1 ms  |
| 1 patch, 32 matches    | 23.4 ms  | 33.8 ms  |
| 16 patches, 32 matches | 24.0 ms  | 28.8 ms  |
| any, 128 matches       | 24-26 ms | 32-43 ms |

(128 matches ties with 32 here only because the fixture has just 32 galaxies to return. Production, with 17,369 to draw from, separates them cleanly.)

Timing the stages of `search()` shows why cost tracks `matches`:

| Stage         | p50         | Share      |
| ------------- | ----------- | ---------- |
| `centroid`    | 0.32 ms     | 1.3 %      |
| `candidates`  | 0.69 ms     | 2.7 %      |
| **`vectors`** | **19.6 ms** | **77.2 %** |
| `score_maps`  | 4.72 ms     | 18.6 %     |
| `rank`        | 0.06 ms     | 0.2 %      |

`vectors` reconstructs 18,432 vectors at **about 1 µs each**, because `reconstruct_batch` walks the IVF direct map one vector at a time: a list lookup and a per-vector decode call rather than a contiguous read.

**Only `vectors` compares across runs.** It is faiss-parallel, and the `score_maps` GEMV contends with that thread pool differently depending on how the process started. Byte-identical work over six fresh processes on one four-core host measured `score_maps` at 0.67, 4.61, 4.89, 5.74, 6.96 and 7.06 ms, against 15.8 to 26.0 ms for `vectors`. So a change to `score_maps` needs the stage isolated before its share means anything. The benchmark reports `environment` rather than pinning threads, since pinning makes `vectors` unrepresentatively slow.

## Cutouts

`cutouts.parquet` holds one PNG per galaxy, read into memory at startup, so serving one is an array lookup: **0.68 ms p50** against the fixture, not yet re-measured against production.

Decoding on demand instead costs 0.43–0.66 s per request against production, 11.3 s on the first request per container, and 11.4 s of wall clock for a 32-row match list, because the work is CPU-bound in one process and `@modal.concurrent` only queues it. That is what the artifact buys, and why no fallback to the old path exists: a fallback would keep the source dataset in the serving image and hide a missing artifact behind an 11-second page.

At a measured ~10.8 KB per production cutout, 17,369 of them are about **190 MB**, and startup grows by that read: under a second at the Modal volume's 330 MB/s, against the index's 47. The build cost is **unmeasured**. PNG encoding is 3.4 ms per cutout, a minute for the corpus, so the job is dominated by decoding the source images, which has not been timed on its own.

## Fixture: metadata endpoints

In-process, no network.

| Endpoint                  | p50     | Note                                   |
| ------------------------- | ------- | -------------------------------------- |
| `/meta`                   | 0.68 ms | cached in-process after the first call |
| `/galaxies/{g}/image.png` | 0.68 ms | array lookup into `cutouts.parquet`    |
| `/galaxies/{g}/tokens`    | 1.67 ms | filtered parquet read                  |
| `/galaxies/{g}/coverage`  | 2.20 ms | filtered parquet read                  |

Against production these sit at the ~240 ms network floor, so their compute is negligible either way.

## Production: artifact sizes

Measured with `HEAD /artifacts/{role}`.

| Artifact          | Size         | Notes                                   |
| ----------------- | ------------ | --------------------------------------- |
| `encoded`         | **22.97 GB** | wired to a "Download embeddings" button |
| `encoded_index`   | **15.50 GB** | read into RAM on every container start  |
| `codebook`        | **2.70 GB**  | same shape as `encoded`, 8.5× smaller   |
| `full_points`     | **180 MB**   | downloaded into the browser on "Full"   |
| `tokens`          | **7.0 MB**   |                                         |
| `parametric_umap` | **923 KB**   | not read at serve time                  |
| `mean_points`     | **259 KB**   | downloaded on first paint               |

The index size matches an arithmetic check: 17,369 × 576 patches = 10,004,544 vectors × 768 dims × 2 bytes for `SQfp16` = 15.37 GB, plus ids and the direct map.

`codebook` is 8.5× smaller than `encoded` despite an identical schema because codebook vectors are drawn from a finite codebook, so the same 768-d rows repeat and zstd compresses them well. Contextualised outputs are all distinct and do not compress.

## Candidate work

Nothing in this section is done. Each entry is triaged on three axes, because they answer different questions and a change can score well on one and badly on another:

- **Impact**: user-visible latency or compute saved, read off the measurements above rather than off intuition.
- **Blast radius**: how much code, and how much of the search method, a change disturbs. The two are independent: a one-line change that alters how every query touches memory has a small diff and a large blast radius.
- **Kind**: _implementation_ pays off directly, _quality_ removes a defect or an obstacle without saving time, _groundwork_ only makes a later change possible or provable.

| Entry                   | Impact                                               | Blast radius                | Kind                     |
| ----------------------- | ---------------------------------------------------- | --------------------------- | ------------------------ |
| `index-mmap`            | 47 s cold start, if faiss supports it here           | Small diff, large behaviour | Implementation           |
| `ssr-unblock`           | Turns 47 s of blank page into 47 s of spinner        | Medium, frontend            | Quality                  |
| `patch-array`           | 7–10× on `/similarity`                               | Large                       | Implementation           |
| `cache-headers`         | Whole surface on repeat visits; magnitude unmeasured | Small                       | Implementation           |
| `index-compression`     | Cold start, and the RAM ceiling                      | Large, methodological       | Implementation           |
| `tokens-bulk`           | ~240 ms per newly selected galaxy                    | Medium                      | Implementation           |
| `bulk-offload`          | None steady-state, large under load                  | Infrastructure              | Quality                  |
| `full-points-view`      | Browser memory and time-to-full-view; unmeasured     | Small-medium                | Implementation           |
| `match-list-virtualise` | Low                                                  | Medium, frontend            | Quality                  |
| `min-containers`        | Removes cold start outright                          | None                        | Operational, costs money |
| `production-recall`     | None directly                                        | Small                       | Groundwork               |
| `recall-reported`       | None                                                 | Small                       | Quality                  |
| `encode-coverage`       | None                                                 | Small                       | Groundwork               |

### Sequencing

Impact order is not the order to work in, because of three couplings.

**`index-mmap` needs measuring before it counts as small.** `write_index` stores an IVF index's inverted lists in the in-memory array form, and faiss's `IO_FLAG_MMAP` may require the on-disk form instead. Even if it loads, `make_direct_map()` plus page faults during `reconstruct_batch` could trade 46s of startup for unpredictable per-query latency. The change is one flag; the question it raises is whether every query gets slower, and only a measurement answers it.

**`patch-array` and `index-compression` are one design, not two changes.** `patch-array` adds a 15.4 GB contiguous fp16 array holding the same vectors the index already holds, against a 64 GB container. `index-compression` shrinks the index at some cost in recall. Done together they separate two jobs the index is currently doing at once: a heavily compressed index generates candidates, and the flat array scores them exactly. Done separately, the first doubles memory and the second loses accuracy for nothing.

**`production-recall` gates both of them.** Production recall is unknown, for the reasons in [Reading fixture numbers](#reading-fixture-numbers), so there is no baseline to show a quantiser change did not silently degrade results.

That gives a working order. `cache-headers` is independent and provable now. Then the `index-mmap` measurement and `production-recall`, which between them make the `patch-array` + `index-compression` design possible.

### Cold start

Measured: 400 s idle, then `GET /meta` took **46.97 s**, against 0.47 s warm. That is `faiss.read_index` reading the whole 15.5 GB index into memory with no mmap, roughly 330 MB/s off the Modal volume, while `max_containers=1` means there is no second container to answer instead.

With `scaledown_window=5*60` this is not a tail case. Any visitor arriving more than five minutes after the last one waits the full 47 s, so on a low-traffic site it is the usual case.

Options, cheapest first:

- **`index-mmap`**: `faiss.IO_FLAG_MMAP` so pages fault in lazily. A search at `nprobe=64` touches ~0.4% of the lists, so time-to-first-response should drop by a large factor even though steady-state stays the same.
- **`ssr-unblock`**: fetch `/meta` client-side so the shell paints immediately and the wait becomes a spinner instead of nothing. The cost is that every consumer of `data.meta` must then handle its absence.
- **`index-compression`**: `SQ8` halves the index, PQ far more, both at some recall cost that `scripts/benchmark.py` can quantify first.
- **`min-containers`**: `min_containers=1` removes cold start entirely, at the price of one container running continuously.

The first two are cheap and compose. Do them before considering the last two.

### No cache headers anywhere

Every response is immutable for a given `DATASET_REVISION`, and not one sets `Cache-Control`. Artifacts carry an `etag` so they revalidate; `/meta`, `/tokens`, `/coverage`, `/similarity` and `/image.png` carry nothing, so every repeat visit re-computes and re-transfers everything.

The one design question in `cache-headers` is what makes the immutability safe to advertise. A long `max-age` with `immutable` is correct only while the revision does not change; a browser that cached under the old revision would keep serving it. Putting the revision in the path or a query parameter makes the whole surface safely `immutable`, after which repeat visits cost nothing and a CDN can serve the traffic.

### Candidate reconstruction

`patch-array` keeps the anchor patches as one contiguous memory-mapped fp16 array beside the index and slices `galaxy * 576 … (galaxy + 1) * 576` out of it. A 33-galaxy result is then 29 MB of sequential reads and one BLAS matmul, which should land under 15 ms, about **7–10×** on this endpoint, taking the 128-match case from ~325 ms to tens of milliseconds.

See [Sequencing](#sequencing) for why it only makes sense alongside `index-compression`.

### Bulk artifacts through the app container

`encoded` and `full_points` both stream through the single serving container, where one download can block every interactive request behind it.

`bulk-offload` puts object storage or a CDN in front. At minimum, a 23 GB download should not be a one-click button on a `max_containers=1` service.

### Frontend

- **`full-points-view`**: `loadParquet` runs `CREATE TABLE … AS SELECT`, so all 180 MB is decoded into WASM memory. A view over the parquet would read ranges on demand instead, but whether that is faster depends on how many queries follow and how well DuckDB-WASM caches those ranges. Unmeasured either way.
- **`match-list-virtualise`**: up to 128 rows × 3 patch grids, each with two canvases. `IsInViewport` gates redraws, which is the only reason this is tolerable, but it caps how far `matches` can usefully go.

### Smaller items

- **`tokens-bulk`**: `/galaxies/{g}/tokens` returns 2.3 KB and costs a ~240 ms round trip, paid once per galaxy the user selects. The whole `tokens` artifact is 7.0 MB, so shipping it once and querying in-browser could remove every one of those round trips. Whether it helps depends on the anchor column's share of that 7 MB, which is unmeasured, and on how many galaxies a session touches.
- **`recall-reported`**: `search()` can return fewer than `matches` galaxies when the candidate patches do not cover enough of them, and the response says nothing about it.
- **`encode-coverage`**: `app/encode.py` has no automated coverage at all, so an import error or a schema drift there is caught only by a full build. It imports torch at module level, so a test needs the stubbed-import approach rather than a real import.

## Reading fixture numbers

The fixture is small, and small changes the index's geometry. Two ratios matter and both are wrong at fixture scale:

|                                  | Fixture (32 galaxies) | Production |
| -------------------------------- | --------------------- | ---------- |
| `nlist`                          | 472                   | 16,384     |
| `nprobe / nlist`                 | 13.6 %                | **0.39 %** |
| `PROBE` as a share of the corpus | 11 %                  | **0.02 %** |

The fixture probes 35× more of its index and its candidate pool covers 11% of all patches. Latency ordering and the stage split carry over; **absolute recall does not**.

Measured recall against brute force over every patch, 16 queries, 31 requested:

| `nprobe` | Recall | Galaxies returned |
| -------- | ------ | ----------------- |
| 1        | 0.70   | 21.6 / 31         |
| 4        | 0.99   | 30.6 / 31         |
| 16 – 472 | 1.00   | 31.0 / 31         |

What this does establish: the metric is defined and wired up, and the "shorter than `matches`" behaviour is real and silent: at `nprobe=1` a third of the requested galaxies never come back, with nothing in the response saying so. **Production recall is unknown** and needs measuring against the real index before any change to `NPROBE`, `PROBE`, `NLIST` or the quantiser.

## Scaling ceilings

Current design targets COSMOS scale. Where it stops:

| Ceiling                      | Now                 | Breaks at                                                       |
| ---------------------------- | ------------------- | --------------------------------------------------------------- |
| Index in RAM                 | 15.5 GB             | ~4×, the container's 64 GB limit. 100× needs PQ or sharding.    |
| Cold start                   | 47 s                | ~2× before it exceeds common proxy and browser timeouts         |
| `full_points` in the browser | 180 MB              | ~10×; DuckDB-WASM has a few GB to work with.                    |
| Cutout precompute            | 190 MB              | linear; fine to ~100×, then needs tiling                        |
| Exact-search reference       | whole corpus in RAM | already fixture-only; `corpus()` cannot run at production scale |
| Serving capacity             | one container       | any concurrency at all; `max_containers=1` is a hard cap        |

None of these need solving now. All of them should be checked before a change assumes they are not there.
