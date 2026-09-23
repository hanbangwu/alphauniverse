# Performance

Every figure here comes from one run of `scripts/benchmark.py`:

| Run              |                                                             |
| ---------------- | ----------------------------------------------------------- |
| Date             | 2026-09-23                                                  |
| Commit           | `b152dd9`                                                   |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                  |
| Server           | `fastapi_app`: 8 CPU, 8 GiB requested, 64 GiB limit         |
| Client           | 1 CPU, separate container                                   |
| Stage container  | server spec; 24 CPUs visible, faiss and OpenMP at 8 threads |
| Warm runs        | 30 per figure, after one discarded warm-up                  |

Latency is measured by the client, so it includes Modal's ingress but no one's home network. Anything labelled **unmeasured** is read off the code and needs a real number before it is trusted.

## The cost model

What a first visitor to an idle site waits for:

| Step                                           | Cost                        |
| ---------------------------------------------- | --------------------------- |
| Container start + index load, blocking `/meta` | **44.9 s**                  |
| Index load: `read_index` plus the direct map   | 40.7 s                      |
| Everything warm after that                     | 0.30–0.64 s p50 per request |

**A cold visit is ~45 seconds of blank page**, because `+layout.server.ts` awaits `/meta` during SSR and nothing renders until it returns.

Cold start dominates, and it is not the search algorithm: the approximate nearest-neighbour search is 14% of a query.

## Query latency

`/similarity`, warm:

| Query shape             | p50    | p95    |
| ----------------------- | ------ | ------ |
| 1 patch, 8 matches      | 340 ms | 351 ms |
| 4 patches, 8 matches    | 336 ms | 342 ms |
| 16 patches, 8 matches   | 339 ms | 351 ms |
| 1 patch, 32 matches     | 415 ms | 439 ms |
| 4 patches, 32 matches   | 397 ms | 416 ms |
| 16 patches, 32 matches  | 404 ms | 422 ms |
| 1 patch, 128 matches    | 645 ms | 750 ms |
| 4 patches, 128 matches  | 612 ms | 661 ms |
| 16 patches, 128 matches | 633 ms | 663 ms |

**Cost is set by `matches` and is independent of how many patches are queried.** The patch count changes one averaging step over a handful of vectors; `matches` changes how many galaxies get fully rescored, at 576 vector reconstructions each. Every request pays the ~305 ms floor `/meta` shows in [Other endpoints](#other-endpoints), so the default of 32 matches adds about 100 ms to it and the UI's maximum of 128 about 320 ms.

## Stages of `search()`

p50 over 30 queries of 4 patches and 32 matches:

| Stage         | p50         | Share      |
| ------------- | ----------- | ---------- |
| `centroid`    | 0.90 ms     | 1.0 %      |
| `candidates`  | 12.8 ms     | 13.9 %     |
| **`vectors`** | **76.8 ms** | **83.2 %** |
| `score_maps`  | 1.50 ms     | 1.6 %      |
| `span_maps`   | 0.09 ms     | 0.1 %      |
| `rank`        | 0.22 ms     | 0.2 %      |
| Total         | 92.3 ms     |            |

`vectors` reconstructs 19,008 vectors at **4.0 µs each**, because `reconstruct_batch` walks the IVF direct map one vector at a time: a list lookup and a per-vector decode call rather than a contiguous read.

Shares depend on the thread configuration: `vectors` is faiss-parallel and the `score_maps` GEMV contends with that pool, so stage figures only compare between runs whose thread configuration agrees.

## Other endpoints

Warm, same client:

| Endpoint                  | p50    | p95    |
| ------------------------- | ------ | ------ |
| `/meta`                   | 305 ms | 313 ms |
| `/galaxies/{g}/image.png` | 303 ms | 310 ms |
| `/galaxies/{g}/tokens`    | 312 ms | 320 ms |
| `/galaxies/{g}/coverage`  | 313 ms | 326 ms |

`/meta` is cached in-process and the image is an array lookup into `cutouts.parquet`, so these sit at the request floor and their compute is negligible next to it. What the floor itself is made of is **unmeasured**.

## Artifact sizes

`Content-Length` of `HEAD /artifacts/{role}`:

| Artifact          | Size         | Notes                                   |
| ----------------- | ------------ | --------------------------------------- |
| `encoded`         | **22.97 GB** | wired to a "Download embeddings" button |
| `encoded_index`   | **17.19 GB** | read into RAM on every container start  |
| `codebook`        | **2.70 GB**  | same shape as `encoded`, 8.5× smaller   |
| `cutouts`         | **215 MB**   | read into memory at startup             |
| `full_points`     | **180 MB**   | downloaded into the browser on "Full"   |
| `spectra`         | **123 MB**   | read into memory at startup             |
| `tokens`          | **7.0 MB**   |                                         |
| `parametric_umap` | **923 KB**   | not read at serve time                  |
| `mean_points`     | **259 KB**   | downloaded on first paint               |

Loading the 17.19 GB index takes 40.7 s. How that splits between `read_index` and `make_direct_map()` is **unmeasured**.

`codebook` is 8.5× smaller than `encoded` despite an identical schema because codebook vectors are drawn from a finite codebook, so the same 768-d rows repeat and zstd compresses them well. Contextualised outputs are all distinct and do not compress.

## Candidate work

Nothing in this section is done. Each entry is triaged on three axes, because they answer different questions and a change can score well on one and badly on another:

- **Impact**: user-visible latency or compute saved, read off the measurements above rather than off intuition.
- **Blast radius**: how much code, and how much of the search method, a change disturbs. The two are independent: a one-line change that alters how every query touches memory has a small diff and a large blast radius.
- **Kind**: _implementation_ pays off directly, _quality_ removes a defect or an obstacle without saving time, _groundwork_ only makes a later change possible or provable.

| Entry                   | Impact                                               | Blast radius                | Kind                     |
| ----------------------- | ---------------------------------------------------- | --------------------------- | ------------------------ |
| `index-mmap`            | 45 s cold start, if faiss supports it here           | Small diff, large behaviour | Implementation           |
| `ssr-unblock`           | Turns 45 s of blank page into 45 s of spinner        | Medium, frontend            | Quality                  |
| `patch-array`           | `vectors`, 83% of a query; magnitude unmeasured      | Large                       | Implementation           |
| `cache-headers`         | Whole surface on repeat visits; magnitude unmeasured | Small                       | Implementation           |
| `index-compression`     | Cold start, and the RAM ceiling                      | Large, methodological       | Implementation           |
| `tokens-bulk`           | ~312 ms per newly selected galaxy                    | Medium                      | Implementation           |
| `bulk-offload`          | None steady-state, large under load                  | Infrastructure              | Quality                  |
| `full-points-view`      | Browser memory and time-to-full-view; unmeasured     | Small-medium                | Implementation           |
| `match-list-virtualise` | Low                                                  | Medium, frontend            | Quality                  |
| `min-containers`        | Removes cold start outright                          | None                        | Operational, costs money |
| `production-recall`     | None directly                                        | Small                       | Groundwork               |
| `recall-reported`       | None                                                 | Small                       | Quality                  |
| `encode-coverage`       | None                                                 | Small                       | Groundwork               |

### Sequencing

Impact order is not the order to work in, because of three couplings.

**`index-mmap` needs measuring before it counts as small.** `write_index` stores an IVF index's inverted lists in the in-memory array form, and faiss's `IO_FLAG_MMAP` may require the on-disk form instead. Even if it loads, `make_direct_map()` plus page faults during `reconstruct_batch` could trade 41 s of startup for unpredictable per-query latency. The change is one flag; the question it raises is whether every query gets slower, and only a measurement answers it.

**`patch-array` and `index-compression` are one design, not two changes.** `patch-array` adds a contiguous fp16 array holding the same patch vectors the index already holds, against a 64 GB container. `index-compression` shrinks the index at some cost in recall. Done together they separate two jobs the index is currently doing at once: a heavily compressed index generates candidates, and the flat array scores them exactly. Done separately, the first doubles memory and the second loses accuracy for nothing.

**`production-recall` gates both of them.** Production recall is **unmeasured**: the brute-force reference in `tests/test_search.py` holds the whole corpus in memory, which only works at fixture scale. There is no baseline to show a quantiser change did not silently degrade results.

That gives a working order. `cache-headers` is independent and provable now. Then the `index-mmap` measurement and `production-recall`, which between them make the `patch-array` + `index-compression` design possible.

### Cold start

A request to a freshly started container took **44.9 s**, against 0.30 s warm. Loading the index is 40.7 s of it: `faiss.read_index` reads the whole 17.19 GB into memory with no mmap, then `make_direct_map()` builds the id lookup, in a split that is unmeasured, while `max_containers=1` means there is no second container to answer instead.

With `scaledown_window=5*60` this is not a tail case. Any visitor arriving more than five minutes after the last one waits the full 45 s, so on a low-traffic site it is the usual case.

Options, cheapest first:

- **`index-mmap`**: `faiss.IO_FLAG_MMAP` so pages fault in lazily. A search at `nprobe=64` of `nlist=16384` touches 0.4% of the lists, so time-to-first-response should drop by a large factor even though steady-state stays the same.
- **`ssr-unblock`**: fetch `/meta` client-side so the shell paints immediately and the wait becomes a spinner instead of nothing. The cost is that every consumer of `data.meta` must then handle its absence.
- **`index-compression`**: `SQ8` halves the index, PQ far more, both at a recall cost that is unmeasured until `production-recall` exists.
- **`min-containers`**: `min_containers=1` removes cold start entirely, at the price of one container running continuously.

The first two are cheap and compose. Do them before considering the last two.

### No cache headers anywhere

Every response is immutable for a given `DATASET_REVISION`, and not one sets `Cache-Control`. Artifacts carry an `etag` so they revalidate; `/meta`, `/tokens`, `/coverage`, `/similarity` and `/image.png` carry nothing, so every repeat visit re-computes and re-transfers everything.

The one design question in `cache-headers` is what makes the immutability safe to advertise. A long `max-age` with `immutable` is correct only while the revision does not change; a browser that cached under the old revision would keep serving it. Putting the revision in the path or a query parameter makes the whole surface safely `immutable`, after which repeat visits cost nothing and a CDN can serve the traffic.

### Candidate reconstruction

`patch-array` keeps the anchor patches as one contiguous memory-mapped fp16 array beside the index and slices `galaxy * 576 … (galaxy + 1) * 576` out of it. A 33-galaxy result is then one sequential read and one BLAS matmul instead of 19,008 direct-map lookups. The speedup is unmeasured.

See [Sequencing](#sequencing) for why it only makes sense alongside `index-compression`.

### Bulk artifacts through the app container

`encoded` and `full_points` both stream through the single serving container, where one download can block every interactive request behind it.

`bulk-offload` puts object storage or a CDN in front. At minimum, a 23 GB download should not be a one-click button on a `max_containers=1` service.

### Frontend

- **`full-points-view`**: `loadParquet` runs `CREATE TABLE … AS SELECT`, so all 180 MB is decoded into WASM memory. A view over the parquet would read ranges on demand instead, but whether that is faster depends on how many queries follow and how well DuckDB-WASM caches those ranges. Unmeasured either way.
- **`match-list-virtualise`**: up to 128 rows × 3 patch grids, each with two canvases. `IsInViewport` gates redraws, which is the only reason this is tolerable, but it caps how far `matches` can usefully go.

### Smaller items

- **`tokens-bulk`**: `/galaxies/{g}/tokens` returns 576 uint32 ids and costs 312 ms p50, paid once per galaxy the user selects. The whole `tokens` artifact is 7.0 MB, so shipping it once and querying in-browser could remove every one of those requests. Whether it helps depends on the anchor column's share of that 7 MB, which is unmeasured, and on how many galaxies a session touches.
- **`recall-reported`**: `search()` can return fewer than `matches` galaxies when the candidate patches do not cover enough of them, and the response says nothing about it.
- **`encode-coverage`**: `app/encode.py` has no automated coverage at all, so an import error or a schema drift there is caught only by a full build. It imports torch at module level, so a test needs the stubbed-import approach rather than a real import.

## Scaling ceilings

Current design targets COSMOS scale. Where it stops:

| Ceiling                      | Now                 | Breaks at                                                      |
| ---------------------------- | ------------------- | -------------------------------------------------------------- |
| Index in RAM                 | 17.19 GB            | ~3.7×, the container's 64 GB limit. 100× needs PQ or sharding. |
| Cold start                   | 44.9 s              | ~2× before it exceeds common proxy and browser timeouts        |
| `full_points` in the browser | 180 MB              | ~10×; DuckDB-WASM has a few GB to work with.                   |
| Cutouts in memory            | 215 MB              | linear; fine to ~100×, then needs tiling                       |
| Exact-search reference       | whole corpus in RAM | already fixture-only; production recall is unmeasured          |
| Serving capacity             | one container       | any concurrency at all; `max_containers=1` is a hard cap       |

None of these need solving now. All of them should be checked before a change assumes they are not there.
