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

Latency is measured by the client, so it includes Modal's ingress and no one's home network. **Unmeasured** marks a figure read off the code.

## The cost model

What a first visitor to an idle site waits for:

| Step                                              | Cost                        |
| ------------------------------------------------- | --------------------------- |
| Container start + startup loads, blocking `/meta` | **19.5 s**                  |
| Startup loads, of which the index is 6.0 s        | 7.9 s                       |
| First `/similarity` after that                    | 0.64 s                      |
| Everything warm after that                        | 0.15–0.49 s p50 per request |

**A cold visit is about 20 s of blank page**: `+layout.server.ts` awaits `/meta` during SSR, and nothing renders until it returns.

Cold start dominates, not the search: the nearest-neighbour search is 22% of `search()`, which is 51 ms of a request.

## Query latency

`/similarity` with 4 patches, warm:

| Matches | p50    | p95    |
| ------- | ------ | ------ |
| 8       | 204 ms | 231 ms |
| 32      | 263 ms | 269 ms |
| 128     | 489 ms | 761 ms |

**`matches` sets the cost**: each match is rescored from 576 reconstructed vectors, while the patch count changes only one averaging step, so the run holds patches at 4. Over the ~148 ms floor every request pays ([Other endpoints](#other-endpoints)), 32 matches add 263 − 148 = 115 ms at p50, and 128 add 489 − 148 = 341 ms.

The first `/similarity` to a fresh container takes **0.64 s**, of which `search()` is 72 ms in the stage container; where the rest goes is **unmeasured**.

## Stages of `search()`

4 patches, 32 matches. The first query runs right after the index loads; warm is the p50 over the 30 that follow:

| Stage         | First query | Warm p50    | Warm share |
| ------------- | ----------- | ----------- | ---------- |
| `centroid`    | 9.80 ms     | 0.65 ms     | 1.3 %      |
| `candidates`  | 11.1 ms     | 11.4 ms     | 22.2 %     |
| **`vectors`** | 42.8 ms     | **38.2 ms** | **74.7 %** |
| `score_maps`  | 1.39 ms     | 0.73 ms     | 1.4 %      |
| `span_maps`   | 6.40 ms     | 0.06 ms     | 0.1 %      |
| `rank`        | 0.19 ms     | 0.16 ms     | 0.3 %      |
| Total         | 71.8 ms     | 51.1 ms     |            |

`vectors` reconstructs 33 × 576 = 19,008 patch vectors in 38.2 ms, **2.0 µs each**: `reconstruct_batch` walks the IVF direct map one vector at a time, a list lookup and a decode call each, not a contiguous read.

`vectors` is faiss-parallel and the `score_maps` GEMV contends with its threads, so stage figures compare only between runs with the same thread configuration.

## Other endpoints

Warm, same client:

| Endpoint                  | p50    | p95    |
| ------------------------- | ------ | ------ |
| `/meta`                   | 148 ms | 153 ms |
| `/galaxies/{g}/image.png` | 148 ms | 157 ms |
| `/galaxies/{g}/tokens`    | 158 ms | 169 ms |
| `/galaxies/{g}/coverage`  | 163 ms | 237 ms |

`/meta` is cached in-process and the image is an array lookup, so these sit at the request floor. What the floor is made of is **unmeasured**.

## Artifact sizes

`Content-Length` of `HEAD /artifacts/{role}`:

| Artifact          | Size         | Notes                                                                                   |
| ----------------- | ------------ | --------------------------------------------------------------------------------------- |
| `encoded`         | **22.97 GB** | wired to a "Download encoded embeddings" button                                         |
| `search_index`    | **17.19 GB** | memory-mapped on every container start                                                  |
| `codebook`        | **2.70 GB**  | same shape as `encoded`, 8.5× smaller; wired to a "Download codebook embeddings" button |
| `images`          | **215 MB**   | read into memory at startup                                                             |
| `full_points`     | **180 MB**   | downloaded into the browser on "Full"                                                   |
| `spectra`         | **123 MB**   | read into memory at startup                                                             |
| `tokens`          | **7.0 MB**   | wired to a "Download tokens" button                                                     |
| `parametric_umap` | **923 KB**   | not read at serve time                                                                  |
| `mean_points`     | **259 KB**   | downloaded on first paint                                                               |

`codebook` is 22.97 / 2.70 = 8.5× smaller than `encoded` with the same schema because its rows depend only on the token id and modality, so they repeat and zstd compresses them. Contextualised outputs are all distinct.

## Cold start

A request to a fresh container took **19.5 s**, against 0.15 s warm. With `max_containers=1` no second container can answer instead.

The startup loads, timed in a separate container of the same spec, add up to 7.9 s:

| Load      | Time   |
| --------- | ------ |
| `index`   | 6.01 s |
| `images`  | 1.37 s |
| `spectra` | 0.44 s |
| `starts`  | 0.03 s |
| `labels`  | 0.01 s |

`faiss.read_index` memory-maps the 17.19 GB index (`IO_FLAG_MMAP`), so pages fault in as queries touch them, and `make_direct_map()` reads every list's ids. The other 19.5 − 7.9 = 11.6 s of a cold request is **unmeasured**.

With `scaledown_window=5*60`, any visitor arriving more than five minutes after the last waits the full 20 s, so on a low-traffic site a cold start is the usual case.

## Scaling ceilings

The design targets COSMOS scale. Where it stops:

| Ceiling                      | Now                 | Breaks at                                                                                                                             |
| ---------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Index size                   | 17.19 GB            | when the pages queries touch outgrow the 64 GiB limit (68.72 GB, or 68.72 / 17.19 = 4.0× the index); how far below that is unmeasured |
| Cold start                   | 19.5 s              | when it exceeds a proxy or browser timeout; which one, and at what length, is unmeasured                                              |
| `full_points` in the browser | 180 MB              | when decoding it outgrows DuckDB-WASM's memory, a limit that is unmeasured                                                            |
| Images in memory             | 215 MB              | grows linearly with the galaxy count; where it breaks is unmeasured                                                                   |
| Exact-search reference       | whole corpus in RAM | when it outgrows the recall job's 128 GiB; its peak memory and production recall are unmeasured until `scripts/recall.py` runs        |
| Serving capacity             | one container       | past 16 concurrent inputs (`max_inputs=16`), unmeasured; `max_containers=1` is a hard cap                                             |

Check these before a change assumes they are not there.
