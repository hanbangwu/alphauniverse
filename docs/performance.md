# Performance

Every figure here comes from one run of `scripts/benchmark.py`, whose report is `docs/benchmarks/latest.json`, except those under [Recall](#recall), which come from one run of `scripts/recall.py`:

| Run              |                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| Date             | 2026-10-02                                                                                                       |
| Commit           | `b3077c5`, both versions                                                                                         |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                       |
| Started by       | the Benchmark workflow, run 37010805704                                                                          |
| Server           | `fastapi_app`: 8 CPU, 8 GiB requested, 64 GiB limit; `max_inputs=16`, `max_containers=1`, `scaledown_window=300` |
| Client           | 1 CPU, separate container; 17 CPUs visible, faiss at 1 thread; AMD family 25, model 1                            |
| Stage container  | server spec; 24 CPUs visible, faiss and OpenMP at 8 threads; AMD family 25, model 1                              |
| Cold runs        | one request, or one query, per figure                                                                            |
| Warm runs        | 30 per figure, after one discarded warm-up                                                                       |
| Stage rounds     | 4, mirrored, two per version                                                                                     |

Latency is measured by the client, so it includes Modal's ingress and no one's home network. **Unmeasured** marks a figure read off the code.

## The cost model

What a first visitor to an idle site waits for:

| Step                                              | Cost                        |
| ------------------------------------------------- | --------------------------- |
| Container start + startup loads, blocking `/meta` | **16.8 s**                  |
| Startup loads, of which the index is 12.5 s       | 14.1 s                      |
| First `/search` after that                        | 0.42 s                      |
| Everything warm after that                        | 0.10–0.45 s p50 per request |

**A cold visit is about 17 s of blank page**: `+layout.server.ts` awaits `/meta` during SSR, and nothing renders until it returns.

Cold start dominates, not the search: the nearest-neighbour search is 21% of `search()`, which is 98 ms of a request.

## Query latency

`/search`, warm, 32 matches unless stated:

| Query                  | p50    | p95    |
| ---------------------- | ------ | ------ |
| 4 patches, 8 matches   | 145 ms | 177 ms |
| 4 patches              | 208 ms | 223 ms |
| 4 patches, 128 matches | 454 ms | 508 ms |
| 4 spans                | 280 ms | 330 ms |
| 4 patches and 4 spans  | 234 ms | 278 ms |

**`matches` sets the cost**: each match is rescored from 576 reconstructed vectors, while the patch count changes only one averaging step, so the run holds patches at 4. Over the ~104 ms floor every request pays ([Other endpoints](#other-endpoints)), 32 matches add 208 − 104 = 104 ms at p50, and 128 add 454 − 104 = 350 ms.

Span queries draw spans a galaxy's DESI spectrum covers, from galaxies that have one.

The first `/search` to a fresh container takes **0.42 s**, of which `search()` is 128 ms in the stage container; where the rest goes is **unmeasured**.

## Stages of `search()`

4 patches, 32 matches. The first query runs right after the index loads, in the first round; warm is the p50 over the 30 that follow, averaged over the version's two rounds:

| Stage         | First query | Warm p50    | Warm share |
| ------------- | ----------- | ----------- | ---------- |
| `centroid`    | 22.1 ms     | 0.88 ms     | 0.9 %      |
| `candidates`  | 17.0 ms     | 20.9 ms     | 21.3 %     |
| **`vectors`** | 72.9 ms     | **74.4 ms** | **76.0 %** |
| `score_maps`  | 3.14 ms     | 1.31 ms     | 1.3 %      |
| `span_maps`   | 12.6 ms     | 0.22 ms     | 0.2 %      |
| `rank`        | 0.26 ms     | 0.20 ms     | 0.2 %      |
| Total         | 128.0 ms    | 97.8 ms     |            |

`vectors` reconstructs 33 × 576 = 19,008 patch vectors in 74.4 ms, **3.9 µs each**: `reconstruct_batch` walks the IVF direct map one vector at a time, a list lookup and a decode call each, not a contiguous read.

`vectors` is faiss-parallel and the `score_maps` GEMV contends with its threads, so stage figures compare only between runs with the same thread configuration.

Both versions ran the same code, so the spread between rounds is noise: the four rounds' warm totals span 96.9–98.8 ms, 1.9 ms or 1.9% of their 98.1 ms mean. A difference between versions smaller than that is within round-to-round variation on this host.

## Recall

From one run of `scripts/recall.py`, whose report is `docs/benchmarks/recall.json`:

| Run              |                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Date             | 2026-10-02                                                                                                                     |
| Commit           | `9b2e182`, the same code as `b3077c5`                                                                                          |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                                     |
| Started by       | the Benchmark workflow, run 37046901562                                                                                        |
| Job              | build image: 16 CPU, 32 GiB requested, 128 GiB limit; 32 CPUs visible, faiss and OpenMP at 16 threads; AMD family 25, model 17 |
| Queries          | 100 per kind, 32 matches, `search()` at `PROBE=2048` and `NPROBE=64`                                                           |

A query's recall is the share of the exact search's 32 galaxies that `search()` also returns:

| Query                 | Mean  | Lowest | All 32 found | Fewer than 32 returned |
| --------------------- | ----- | ------ | ------------ | ---------------------- |
| 4 patches             | 98.0% | 65.6%  | 74%          | 0%                     |
| 4 spans               | 96.8% | 71.9%  | 67%          | 0%                     |
| 4 patches and 4 spans | 94.8% | 68.8%  | 52%          | 2%                     |

Galaxies `search()` does not return count as misses, so the 2% of combined queries that came back short also lose recall.

Loading every patch and span embedding took 71.8 s of the 171.3 s run, and the job's memory peaked at **86.93 GB**.

## Other endpoints

Warm, same client:

| Endpoint                      | p50    | p95    |
| ----------------------------- | ------ | ------ |
| `/meta`                       | 104 ms | 120 ms |
| `/galaxy/{g}/image`           | 102 ms | 120 ms |
| `/galaxy/{g}/image/tokens`    | 108 ms | 117 ms |
| `/galaxy/{g}`                 | 109 ms | 125 ms |
| `/galaxy/{g}/spectrum`        | 108 ms | 128 ms |
| `/galaxy/{g}/spectrum/tokens` | 109 ms | 127 ms |

`/meta` is built from cached labels and the image is an array lookup, so these sit at the request floor. What the floor is made of is **unmeasured**.

## Concurrency

Each client is a thread with its own connection, sending `/search` with 4 patches at 32 matches:

| Clients | p50    | p95    | Requests/s |
| ------- | ------ | ------ | ---------- |
| 1       | 228 ms | 312 ms | 4.17       |
| 4       | 382 ms | 500 ms | 10.34      |
| 16      | 1.51 s | 2.14 s | 10.63      |
| 32      | 2.91 s | 3.86 s | 10.77      |

**Throughput stops rising at about 10.5 requests/s**, from 4 clients up, and latency grows with the queue: 32 clients / 10.77 requests/s = 2.97 s, against a p50 of 2.91 s. Modal's request log for the run shows server execution rising to 0.45–1.9 s per `/search` at 32 clients, so the server, not the 1-CPU client, sets the ceiling.

## Artifact sizes

`Content-Length` of `HEAD /artifacts/{role}`, which served every role when this was measured; the benchmark now measures only `/projections/{projection}` and `/downloads/{role}`:

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

A request to a fresh container took **16.8 s**, against 0.10 s warm. With `max_containers=1` no second container can answer instead.

The startup loads, timed in a separate container of the same spec, add up to 14.1 s in its first round:

| Load      | First round | Later rounds |
| --------- | ----------- | ------------ |
| `index`   | 12.45 s     | 0.16–0.18 s  |
| `spectra` | 0.78 s      | 0.45–0.46 s  |
| `images`  | 0.60 s      | 0.18–0.19 s  |
| `starts`  | 0.17 s      | 0.01 s       |
| `labels`  | 0.09 s      | 0.01 s       |
| Total     | 14.09 s     | 0.83–0.84 s  |

Later rounds run in the same container, so the files are already in its page cache. `faiss.read_index` memory-maps the 17.19 GB index (`IO_FLAG_MMAP`), so pages fault in as queries touch them, and `make_direct_map()` reads every list's ids. The other 16.8 − 14.1 = 2.7 s of a cold request is **unmeasured**.

With `scaledown_window=300`, any visitor arriving more than five minutes after the last waits the full 17 s, so on a low-traffic site a cold start is the usual case.

## Scaling ceilings

The design targets COSMOS scale. Where it stops:

| Ceiling                      | Now           | Breaks at                                                                                                                             |
| ---------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Index size                   | 17.19 GB      | when the pages queries touch outgrow the 64 GiB limit (68.72 GB, or 68.72 / 17.19 = 4.0× the index); how far below that is unmeasured |
| Cold start                   | 16.8 s        | when it exceeds a proxy or browser timeout; which one, and at what length, is unmeasured                                              |
| `full_points` in the browser | 180 MB        | when decoding it outgrows DuckDB-WASM's memory, a limit that is unmeasured                                                            |
| Images in memory             | 215 MB        | grows linearly with the galaxy count; where it breaks is unmeasured                                                                   |
| Exact-search reference       | 86.93 GB peak | when it outgrows the recall job's 128 GiB (137.44 GB); it uses 86.93 / 137.44 = 63% of that                                           |
| Serving capacity             | one container | `/search` levels off at about 10.5 requests/s from 4 concurrent clients; `max_containers=1` is a hard cap                             |

Check these before a change assumes they are not there.
