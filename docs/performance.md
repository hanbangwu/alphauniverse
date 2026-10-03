# Performance

Every figure here comes from one run of `scripts/benchmark.py`, whose report is `docs/benchmarks/latest.json`, except those under [Recall](#recall), which come from one run of `scripts/recall.py`:

| Run              |                                                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------- |
| Date             | 2026-10-03                                                                                                           |
| Commits          | `676c560` (after, #127), `00b96f8` (before, `main` when the run started), `b3077c5` (best, from the previous report) |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                           |
| Started by       | the Benchmark workflow, run 37142850812                                                                              |
| Server           | `fastapi_app`: 8 CPU, 8 GiB requested, 32 GiB limit; `max_inputs=16`, `max_containers=1`, `scaledown_window=300`     |
| Client           | 1 CPU, separate container; 17 CPUs visible, faiss at 1 thread; Intel family 6, model 85                              |
| Stage container  | server spec; 24 CPUs visible, faiss and OpenMP at 8 threads; AMD family 25, model 17                                 |
| Cold runs        | one request, or one query, per figure                                                                                |
| Warm runs        | 30 per figure, after one discarded warm-up                                                                           |
| Stage rounds     | 6, mirrored, two per version                                                                                         |

Request latency is timed for the after version only, and every other figure below is also the after version's unless it names another.

Latency is measured by the client, so it includes Modal's ingress and no one's home network. **Unmeasured** marks a figure read off the code.

## The cost model

What a first visitor to an idle site waits for:

| Step                                      | Cost                        |
| ----------------------------------------- | --------------------------- |
| Container start + loads, blocking `/meta` | **13.3 s**                  |
| Loads, of which the index is 6.3 s        | 11.6 s                      |
| First `/search` after that                | 0.20 s                      |
| Everything warm after that                | 0.05–0.38 s p50 per request |

**A cold visit is about 13 s of blank page**: `+layout.server.ts` awaits `/meta` during SSR, and nothing renders until it returns.

Cold start dominates, not the search: the nearest-neighbour search is 19% of `search()`, which is 95 ms of a 149 ms `/search`.

## Query latency

`/search`, warm, 32 matches unless stated:

| Query                  | p50    | p95    |
| ---------------------- | ------ | ------ |
| 4 patches, 8 matches   | 95 ms  | 108 ms |
| 4 patches              | 149 ms | 163 ms |
| 4 patches, 128 matches | 379 ms | 422 ms |
| 4 spans                | 190 ms | 196 ms |
| 4 patches and 4 spans  | 158 ms | 185 ms |

**`matches` sets the cost**: each match is rescored from 576 reconstructed vectors, while the patch count changes only one averaging step, so the run holds patches at 4. Over the ~48 ms floor every request pays ([Other endpoints](#other-endpoints)), 32 matches add 149 − 48 = 101 ms at p50, and 128 add 379 − 48 = 331 ms.

Span queries draw spans a galaxy's DESI spectrum covers, from galaxies that have one.

The first `/search` to a fresh container takes **0.20 s**, of which `search()` is 130 ms in the stage container; where the rest goes is **unmeasured**.

## Stages of `search()`

4 patches, 32 matches. The first query runs right after the index loads, in the version's first round; warm is the p50 over the 30 that follow, averaged over the version's two rounds:

| Stage         | First query | Warm p50    | Warm share |
| ------------- | ----------- | ----------- | ---------- |
| `centroid`    | 1.06 ms     | 0.27 ms     | 0.3 %      |
| `candidates`  | 25.5 ms     | 18.1 ms     | 19.2 %     |
| **`vectors`** | 87.3 ms     | **75.0 ms** | **79.3 %** |
| `score_maps`  | 2.04 ms     | 0.91 ms     | 1.0 %      |
| `span_maps`   | 13.5 ms     | 0.20 ms     | 0.2 %      |
| `rank`        | 0.26 ms     | 0.09 ms     | 0.1 %      |
| Total         | 129.6 ms    | 94.6 ms     |            |

`vectors` reconstructs 33 × 576 = 19,008 patch vectors in 75.0 ms, **3.9 µs each**: `reconstruct_batch` walks the IVF direct map one vector at a time, a list lookup and a decode call each, not a contiguous read.

`vectors` is faiss-parallel and the `score_maps` GEMV contends with its threads, so stage figures compare only between runs with the same thread configuration.

The three versions' mean warm totals are 94.6 ms (after), 95.7 ms (before) and 91.4 ms (best), within 95.7 − 91.4 = 4.3 ms of each other, while one version's two rounds differ by up to 100.0 − 89.1 = 10.9 ms (after). On this host no version is measurably faster than another.

## Recall

From one run of `scripts/recall.py`, whose report is `docs/benchmarks/recall.json`:

| Run              |                                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Date             | 2026-10-03                                                                                                                     |
| Commit           | `1dec658`                                                                                                                      |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                                     |
| Started by       | the Benchmark workflow, run 37131608378                                                                                        |
| Job              | build image: 16 CPU, 32 GiB requested, 128 GiB limit; 32 CPUs visible, faiss and OpenMP at 16 threads; AMD family 25, model 17 |
| Queries          | 100 per kind, 32 matches, `search()` at `PROBE=2048` and `NPROBE=64`                                                           |

The first kind draws its galaxies from all galaxies. The other three share one draw of 100 galaxies with a DESI spectrum and the same 4 patches and 4 spans per galaxy, so they differ only in the query's modality.

A query's recall is the share of the exact search's 32 galaxies that `search()` also returns; galaxies `search()` does not return count as misses:

| Query                 | Drawn from | Mean  | Lowest | All 32 found | Fewer than 32 returned |
| --------------------- | ---------- | ----- | ------ | ------------ | ---------------------- |
| 4 patches             | all        | 98.0% | 65.6%  | 74%          | 0%                     |
| 4 patches             | DESI       | 96.7% | 71.9%  | 63%          | 0%                     |
| 4 spans               | DESI       | 96.7% | 53.1%  | 70%          | 0%                     |
| 4 patches and 4 spans | DESI       | 94.3% | 43.8%  | 47%          | 0%                     |

Loading every patch and span embedding took 67.1 s of the 211.9 s run, and the job's memory peaked at **86.84 GB**.

### What the matches hold

Of the 17,369 galaxies, 4,007 (23.1%) have a DESI spectrum, 29 (0.2%) only an SDSS spectrum, and 13,333 (76.8%) neither. Each kind's 100 queries return 3,200 matches. A DESI match is won by a span when its best span scores above its best patch:

| Query                 | Drawn from | With DESI     | SDSS only | Neither       | DESI matches won by a span |
| --------------------- | ---------- | ------------- | --------- | ------------- | -------------------------- |
| 4 patches             | all        | 413 (12.9%)   | 0         | 2,787 (87.1%) | 0 of 413                   |
| 4 patches             | DESI       | 1,221 (38.2%) | 5 (0.2%)  | 1,974 (61.7%) | 0 of 1,221                 |
| 4 spans               | DESI       | 3,126 (97.7%) | 0         | 74 (2.3%)     | 3,071 of 3,126 (98.2%)     |
| 4 patches and 4 spans | DESI       | 1,337 (41.8%) | 1 (0.03%) | 1,862 (58.2%) | 499 of 1,337 (37.3%)       |

- **Span queries return galaxies with spectra and match them on spans**: 97.7% of their matches have a DESI spectrum, against 38.2% for patch queries from the same galaxies, and 98.2% of those are won by a span. 2.3% of their matches have no spectrum.
- **No patch query's match is won by a span**: 0 of 1,634 DESI matches (413 + 1,221).
- **Combined queries match much as patch queries do** (41.8% with DESI, against 38.2%), and spans win 37.3% of their DESI matches.
- Patch queries from DESI galaxies return DESI galaxies at 38.2%, against 12.9% from all galaxies, so kinds compare only within the paired draw.
- Every winning span lies inside its spectrum's observed wavelength range (3,071 of 3,071 and 499 of 499): in these queries no span encoded without data wins a match.
- SDSS-only matches stay at or below their 0.2% share of all galaxies (at most 5 of 3,200): in these queries the all-zero SDSS encodings (#117) draw no more matches than their share.

## Other endpoints

Warm, same client:

| Endpoint                      | p50   | p95   |
| ----------------------------- | ----- | ----- |
| `/meta`                       | 48 ms | 53 ms |
| `/galaxy/{g}/image`           | 50 ms | 59 ms |
| `/galaxy/{g}/image/tokens`    | 54 ms | 59 ms |
| `/galaxy/{g}`                 | 55 ms | 70 ms |
| `/galaxy/{g}/spectrum`        | 58 ms | 65 ms |
| `/galaxy/{g}/spectrum/tokens` | 55 ms | 58 ms |

`/meta` reads cached labels, and the image and spectrum routes read their row from the dataset on each request; all six sit within 58 − 48 = 10 ms of each other at p50. What the floor is made of is **unmeasured**.

## Concurrency

Each client is a thread with its own connection, sending `/search` with 4 patches at 32 matches:

| Clients | p50    | p95    | Requests/s |
| ------- | ------ | ------ | ---------- |
| 1       | 150 ms | 169 ms | 6.58       |
| 4       | 296 ms | 360 ms | 13.48      |
| 16      | 1.35 s | 1.84 s | 11.88      |
| 32      | 2.70 s | 3.28 s | 11.76      |

**Throughput peaks at 13.48 requests/s with 4 clients** and holds near 11.8 from 16 up, and latency grows with the queue: 32 clients / 11.76 requests/s = 2.72 s, against a p50 of 2.70 s. Whether the server or the 1-CPU client sets the ceiling is **unmeasured** in this run.

## Artifact sizes

`Content-Length` of `HEAD /projections/{projection}` and `HEAD /downloads/{role}`:

| Artifact          | Size           | Notes                                                                                   |
| ----------------- | -------------- | --------------------------------------------------------------------------------------- |
| `encoded`         | **22.97 GB**   | wired to a "Download encoded embeddings" button                                         |
| `codebook`        | **2.70 GB**    | same shape as `encoded`, 8.5× smaller; wired to a "Download codebook embeddings" button |
| `full_points`     | **180 MB**     | downloaded into the browser on "Full"                                                   |
| `tokens`          | **7.0 MB**     | wired to a "Download tokens" button                                                     |
| `mean_points`     | **259 KB**     | downloaded on first paint                                                               |
| `search_index`    | **unmeasured** | not served; memory-mapped on every container start                                      |
| `parametric_umap` | **unmeasured** | not served or read at serve time                                                        |

`codebook` is 22.97 / 2.70 = 8.5× smaller than `encoded` with the same schema because its rows depend only on the token id and modality, so they repeat and zstd compresses them. Contextualised outputs are all distinct.

## Cold start

A request to a fresh container took **13.3 s**, against 0.05 s warm. With `max_containers=1` no second container can answer instead.

`lifespan` loads `index` and `starts`, and the first `/meta` loads `dataset` and `labels`. Timed in a separate container of the same spec, they add up to 11.6 s in the version's first round:

| Load      | First round | Later rounds  |
| --------- | ----------- | ------------- |
| `index`   | 6.31 s      | 0.17–0.19 s   |
| `dataset` | 3.99 s      | 0.43–0.55 s   |
| `labels`  | 1.25 s      | 0.002–0.036 s |
| `starts`  | 0.04 s      | 0.02 s        |
| Total     | 11.59 s     | 0.65–0.76 s   |

Later rounds are the other three rounds of the two versions that read the dataset, run in the same container, so the files are already in its page cache. `faiss.read_index` memory-maps the index (`IO_FLAG_MMAP`), so pages fault in as queries touch them, and `make_direct_map()` reads every list's ids. The other 13.3 − 11.6 = 1.7 s of a cold request is **unmeasured**.

With `scaledown_window=300`, any visitor arriving more than five minutes after the last waits the full 13 s, so on a low-traffic site a cold start is the usual case.

## Scaling ceilings

The design targets COSMOS scale. Where it stops:

| Ceiling                      | Now           | Breaks at                                                                                                  |
| ---------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| Index size                   | unmeasured    | when the pages queries touch outgrow the 32 GiB limit (34.36 GB); how far below that is unmeasured         |
| Cold start                   | 13.3 s        | when it exceeds a proxy or browser timeout; which one, and at what length, is unmeasured                   |
| `full_points` in the browser | 180 MB        | when decoding it outgrows DuckDB-WASM's memory, a limit that is unmeasured                                 |
| Exact-search reference       | 86.84 GB peak | when it outgrows the recall job's 128 GiB (137.44 GB); it uses 86.84 / 137.44 = 63% of that                |
| Serving capacity             | one container | `/search` throughput peaks at 13.48 requests/s with 4 concurrent clients; `max_containers=1` is a hard cap |

Check these before a change assumes they are not there.
