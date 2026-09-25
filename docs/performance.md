# Performance

Every figure here comes from one run of `scripts/benchmark.py`:

| Run              |                                                             |
| ---------------- | ----------------------------------------------------------- |
| Date             | 2026-09-24                                                  |
| Commit           | `1d1d4b8`                                                   |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                  |
| Server           | `fastapi_app`: 8 CPU, 8 GiB requested, 64 GiB limit         |
| Client           | 1 CPU, separate container                                   |
| Stage container  | server spec; 24 CPUs visible, faiss and OpenMP at 8 threads |
| Cold runs        | one request, or one query, per figure                       |
| Warm runs        | 30 per figure, after one discarded warm-up                  |

Latency is measured by the client, so it includes Modal's ingress but no one's home network. Anything labelled **unmeasured** is read off the code and needs a real number before it is trusted.

## The cost model

What a first visitor to an idle site waits for:

| Step                                              | Cost                        |
| ------------------------------------------------- | --------------------------- |
| Container start + startup loads, blocking `/meta` | **19.5 s**                  |
| Startup loads, of which the index is 6.0 s        | 7.9 s                       |
| First `/similarity` after that                    | 0.64 s                      |
| Everything warm after that                        | 0.15–0.49 s p50 per request |

**A cold visit is ~20 seconds of blank page**, because `+layout.server.ts` awaits `/meta` during SSR and nothing renders until it returns.

Cold start dominates, and it is not the search algorithm: the approximate nearest-neighbour search is 22% of `search()`, which is itself 51 ms of a request.

## Query latency

`/similarity` with 4 patches, warm:

| Matches | p50    | p95    |
| ------- | ------ | ------ |
| 8       | 204 ms | 231 ms |
| 32      | 263 ms | 269 ms |
| 128     | 489 ms | 761 ms |

**Cost is set by `matches`.** It sets how many galaxies get fully rescored, at 576 vector reconstructions each, while the patch count only changes one averaging step over a handful of vectors, so the run holds patches at 4. Every request pays the ~148 ms floor `/meta` shows in [Other endpoints](#other-endpoints), so the default of 32 matches adds about 115 ms to it and the UI's maximum of 128 about 341 ms.

The first `/similarity` to a freshly started container takes **0.64 s**. `search()` is 72 ms of a first query in the stage container, so most of the rest is outside it, in a split that is **unmeasured**.

## Stages of `search()`

Queries of 4 patches and 32 matches. The first query runs right after the index loads; warm is the p50 over the 30 that follow it:

| Stage         | First query | Warm p50    | Warm share |
| ------------- | ----------- | ----------- | ---------- |
| `centroid`    | 9.80 ms     | 0.65 ms     | 1.3 %      |
| `candidates`  | 11.1 ms     | 11.4 ms     | 22.2 %     |
| **`vectors`** | 42.8 ms     | **38.2 ms** | **74.7 %** |
| `score_maps`  | 1.39 ms     | 0.73 ms     | 1.4 %      |
| `span_maps`   | 6.40 ms     | 0.06 ms     | 0.1 %      |
| `rank`        | 0.19 ms     | 0.16 ms     | 0.3 %      |
| Total         | 71.8 ms     | 51.1 ms     |            |

`vectors` reconstructs 19,008 vectors at **2.0 µs each**, because `reconstruct_batch` walks the IVF direct map one vector at a time: a list lookup and a per-vector decode call rather than a contiguous read.

Shares depend on the thread configuration: `vectors` is faiss-parallel and the `score_maps` GEMV contends with that pool, so stage figures only compare between runs whose thread configuration agrees.

## Other endpoints

Warm, same client:

| Endpoint                  | p50    | p95    |
| ------------------------- | ------ | ------ |
| `/meta`                   | 148 ms | 153 ms |
| `/galaxies/{g}/image.png` | 148 ms | 157 ms |
| `/galaxies/{g}/tokens`    | 158 ms | 169 ms |
| `/galaxies/{g}/coverage`  | 163 ms | 237 ms |

`/meta` is cached in-process and the image is an array lookup into `cutouts.parquet`, so these sit at the request floor and their compute is negligible next to it. What the floor itself is made of is **unmeasured**.

## Artifact sizes

`Content-Length` of `HEAD /artifacts/{role}`:

| Artifact          | Size         | Notes                                   |
| ----------------- | ------------ | --------------------------------------- |
| `encoded`         | **22.97 GB** | wired to a "Download embeddings" button |
| `encoded_index`   | **17.19 GB** | memory-mapped on every container start  |
| `codebook`        | **2.70 GB**  | same shape as `encoded`, 8.5× smaller   |
| `cutouts`         | **215 MB**   | read into memory at startup             |
| `full_points`     | **180 MB**   | downloaded into the browser on "Full"   |
| `spectra`         | **123 MB**   | read into memory at startup             |
| `tokens`          | **7.0 MB**   |                                         |
| `parametric_umap` | **923 KB**   | not read at serve time                  |
| `mean_points`     | **259 KB**   | downloaded on first paint               |

Loading the 17.19 GB index takes 6.0 s. `read_index` memory-maps its inverted lists and `make_direct_map()` reads every list's ids.

`codebook` is 8.5× smaller than `encoded` despite an identical schema because each of its rows depends only on the token id and modality, so the same 768-d rows repeat and zstd compresses them well. Contextualised outputs are all distinct and do not compress.

## Candidate work

Nothing in this section is done. Each entry is triaged on three axes, because they answer different questions and a change can score well on one and badly on another:

- **Impact**: user-visible latency or compute saved, read off the measurements above rather than off intuition.
- **Blast radius**: how much code, and how much of the search method, a change disturbs. The two are independent: a one-line change that alters how every query touches memory has a small diff and a large blast radius.
- **Kind**: _implementation_ pays off directly, _quality_ removes a defect or an obstacle without saving time, _groundwork_ only makes a later change possible or provable.

| Entry                   | Impact                                               | Blast radius          | Kind                     |
| ----------------------- | ---------------------------------------------------- | --------------------- | ------------------------ |
| `ssr-unblock`           | Turns the cold start's blank page into a spinner     | Medium, frontend      | Quality                  |
| `patch-array`           | `vectors`, 75% of a query; magnitude unmeasured      | Large                 | Implementation           |
| `cache-headers`         | Whole surface on repeat visits; magnitude unmeasured | Small                 | Implementation           |
| `index-compression`     | Cold start, and the RAM ceiling                      | Large, methodological | Implementation           |
| `tokens-bulk`           | ~158 ms per newly selected galaxy                    | Medium                | Implementation           |
| `bulk-offload`          | None steady-state, large under load                  | Infrastructure        | Quality                  |
| `full-points-view`      | Browser memory and time-to-full-view; unmeasured     | Small-medium          | Implementation           |
| `match-list-virtualise` | Low                                                  | Medium, frontend      | Quality                  |
| `min-containers`        | Removes cold start outright                          | None                  | Operational, costs money |
| `production-recall`     | None directly                                        | Small                 | Groundwork               |
| `recall-reported`       | None                                                 | Small                 | Quality                  |
| `encode-coverage`       | None                                                 | Small                 | Groundwork               |

### Sequencing

Impact order is not the order to work in, because of two couplings.

**`patch-array` and `index-compression` are one design, not two changes.** `patch-array` adds a contiguous fp16 array holding the same patch vectors the index already holds, against a 64 GB container. `index-compression` shrinks the index at some cost in recall. Done together they separate two jobs the index is currently doing at once: a heavily compressed index generates candidates, and the flat array scores them exactly. Done separately, the first doubles memory and the second loses accuracy for nothing.

**`production-recall` gates both of them.** Production recall is **unmeasured**: the brute-force reference in `tests/test_search.py` holds the whole corpus in memory, which only works at fixture scale. There is no baseline to show a quantiser change did not silently degrade results.

That gives a working order. `cache-headers` is independent and provable now. Then `production-recall`, which makes the `patch-array` + `index-compression` design possible.

### Cold start

A request to a freshly started container took **19.5 s**, against 0.15 s warm. `max_containers=1` means there is no second container to answer instead.

The startup loads, timed in a separate container of the same spec, add up to 7.9 s:

| Load      | Time   |
| --------- | ------ |
| `index`   | 6.01 s |
| `cutouts` | 1.37 s |
| `spectra` | 0.44 s |
| `starts`  | 0.03 s |
| `labels`  | 0.01 s |

`faiss.read_index` memory-maps the 17.19 GB with `IO_FLAG_MMAP`, so pages fault in as queries touch them, and `make_direct_map()` reads every id. What the other 11.6 s of a cold request is made of is **unmeasured**.

With `scaledown_window=5*60` this is not a tail case. Any visitor arriving more than five minutes after the last one waits the full 20 s, so on a low-traffic site it is the usual case.

Options, cheapest first:

- **`ssr-unblock`**: fetch `/meta` client-side so the shell paints immediately and the wait becomes a spinner instead of nothing. The cost is that every consumer of `data.meta` must then handle its absence.
- **`index-compression`**: `SQ8` halves the index, PQ far more, both at a recall cost that is unmeasured until `production-recall` exists.
- **`min-containers`**: `min_containers=1` removes cold start entirely, at the price of one container running continuously.

`ssr-unblock` is cheap. Do it before considering the other two.

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
- **`match-list-virtualise`**: up to 128 rows × 3 ECharts instances (two patch grids and a spectrum), all initialised and rendered whether on screen or not, which caps how far `matches` can usefully go.

### Smaller items

- **`tokens-bulk`**: `/galaxies/{g}/tokens` returns 576 uint32 ids and costs 158 ms p50, paid once per galaxy the user selects. The whole `tokens` artifact is 7.0 MB, so shipping it once and querying in-browser could remove every one of those requests. Whether it helps depends on the anchor column's share of that 7 MB, which is unmeasured, and on how many galaxies a session touches.
- **`recall-reported`**: `search()` can return fewer than `matches` galaxies when the candidate patches do not cover enough of them, and the response says nothing about it.
- **`encode-coverage`**: `app/encode.py` has no automated coverage at all, so an import error or a schema drift there is caught only by a full build. It imports torch at module level, so a test needs the stubbed-import approach rather than a real import.

## Scaling ceilings

Current design targets COSMOS scale. Where it stops:

| Ceiling                      | Now                 | Breaks at                                                                                     |
| ---------------------------- | ------------------- | --------------------------------------------------------------------------------------------- |
| Index size                   | 17.19 GB            | ~3.7×, when the pages queries touch outgrow the container's 64 GB. 100× needs PQ or sharding. |
| Cold start                   | 19.5 s              | ~4× before it exceeds common proxy and browser timeouts                                       |
| `full_points` in the browser | 180 MB              | ~10×; DuckDB-WASM has a few GB to work with.                                                  |
| Cutouts in memory            | 215 MB              | linear; fine to ~100×, then needs tiling                                                      |
| Exact-search reference       | whole corpus in RAM | already fixture-only; production recall is unmeasured                                         |
| Serving capacity             | one container       | past 16 concurrent inputs (`max_inputs=16`), unmeasured; `max_containers=1` is a hard cap     |

None of these need solving now. All of them should be checked before a change assumes they are not there.
