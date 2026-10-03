# Benchmarks

Every benchmark runs on Modal and measures the production artifacts on the volume.

| Script                | Measures                                     | Runs on                                   | Report                                     |
| --------------------- | -------------------------------------------- | ----------------------------------------- | ------------------------------------------ |
| `search_performance`  | loads and `search()` stages, across commits  | Modal, a container with the server's spec | `docs/benchmarks/search_performance.json`  |
| `backend_performance` | HTTP latency, cold start and concurrency     | Modal, an ephemeral server and a client   | `docs/benchmarks/backend_performance.json` |
| `search_quality`      | recall of `search()` against an exact search | Modal, a build-image container            | `docs/benchmarks/search_quality.json`      |

```sh
uv run modal run -m scripts.benchmarks.search_performance   # --runs, default 30
uv run modal run -m scripts.benchmarks.backend_performance  # --runs, default 30
uv run modal run -m scripts.benchmarks.search_quality       # --per-kind, default 100
```

- The performance scripts refuse to run with uncommitted changes.
- Each run replaces its report; commit it before the next run.
- Figures compare only within one run: not across runs, machines or thread layouts.
- **Unmeasured** marks a figure read off the code.

## From GitHub Actions

Start the Benchmark workflow by hand from the Actions tab. Pick a branch that contains the workflow, a script, and optionally `runs` for a performance script or `per_kind` for `search_quality` (empty keeps the default). The job uses the deploy job's Modal token. The report appears in the run's summary and as its artifact; the job does not commit it. A job past six hours stops and leaves no report.

## Cost model

What a first visitor to an idle site waits for, from the last `backend_performance` run:

| Step                             | Cost                        |
| -------------------------------- | --------------------------- |
| Container start + loads, `/meta` | **19.0 s**                  |
| First `/search` after that       | 0.23 s                      |
| Everything warm after that       | 0.05–0.40 s p50 per request |

**A cold visit is about 19 s of blank page**: `+layout.server.ts` awaits `/meta` during SSR. Cold start dominates, not the search.

## `search_performance`

A container with the server's spec runs, per version:

1. the loads a cold `/meta` waits for: `dataset`, `labels`, `index`, `starts`;
2. the stages of `search()`, first query and warm, at 4 patches and 32 matches;
3. `search()` whole at 8, 32 and 128 matches.

Versions are the checked-out commit (after), `origin/main` (before), and the stored report's best unless its code matches one of those. Each is a `git archive` of `app/`, `scripts/` and `modal_app.py`, run in a subprocess with after's locked dependencies, in mirrored order: after, before, best, then back. After's first round has the cold page cache. A failed round records its error.

The best version has the lowest `total_p50_ms` (sum of warm stage medians at 32 matches) averaged over its rounds, among versions whose rounds all succeeded. A stored best missing from the clone is dropped with a note. If no version succeeds, best keeps the stored commit without its figure.

### Last run

| Run              |                                                                                                               |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| Date             | 2026-10-03                                                                                                    |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                    |
| Versions         | `ff2ee6a` (after), `c382995` (before, `main`); no stored best                                                 |
| Container        | 8 CPU, 8 GiB requested, 32 GiB limit; 24 CPUs visible, faiss and OpenMP at 8 threads; AMD family 25, model 17 |
| Rounds           | 4, two per version; 30 warm queries per figure after one discarded                                            |

Figures are after's. Warm figures average its two rounds.

**Loads.** Later rounds read files already in the page cache:

| Load      | First round | Later rounds |
| --------- | ----------- | ------------ |
| `index`   | 5.68 s      | 0.15–0.17 s  |
| `labels`  | 1.19 s      | 0.002 s      |
| `dataset` | 0.62 s      | 0.36–0.40 s  |
| `starts`  | 0.02 s      | 0.02 s       |
| Total     | 7.51 s      | 0.54–0.58 s  |

`faiss.read_index` memory-maps the index, so pages fault in as queries touch them; `make_direct_map()` reads every list's ids.

**Stages**, 4 patches, 32 matches:

| Stage         | First query | Warm p50    | Warm share |
| ------------- | ----------- | ----------- | ---------- |
| `centroid`    | 15.7 ms     | 0.79 ms     | 0.8 %      |
| `candidates`  | 14.0 ms     | 17.6 ms     | 17.8 %     |
| **`vectors`** | 82.6 ms     | **79.6 ms** | **80.5 %** |
| `score_maps`  | 1.30 ms     | 0.63 ms     | 0.6 %      |
| `span_maps`   | 11.5 ms     | 0.18 ms     | 0.2 %      |
| `rank`        | 0.12 ms     | 0.09 ms     | 0.1 %      |
| Total         | 125.1 ms    | 98.9 ms     |            |

- **`vectors` dominates**: it reconstructs 33 × 576 = 19,008 patch vectors in 79.6 ms, 4.2 µs each. `reconstruct_batch` walks the IVF direct map one vector at a time, not a contiguous read.
- `vectors` is faiss-parallel and `score_maps` contends with its threads, so stage figures compare only at the same thread configuration.

**Whole `search()`**, 4 patches, warm p50:

| Matches | p50      |
| ------- | -------- |
| 8       | 41.0 ms  |
| 32      | 100.3 ms |
| 128     | 318.5 ms |

**Versions.** Mean warm totals are 98.9 ms (after) and 99.5 ms (before), 0.6 ms apart; after's two rounds differ by 100.76 − 97.05 = 3.71 ms. Neither version is measurably faster.

## `backend_performance`

`modal run` starts an ephemeral `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency. A 1-CPU client in a separate container times, in order:

1. one cold `/meta` and one cold `/search`;
2. warm, `runs` times each after one warm-up: `/meta`, image, image tokens, galaxy, and `/search` with 4 patches at 8, 32 and 128 matches;
3. warm: both spectrum routes, and `/search` at 32 matches with 4 spans, alone and with 4 patches, on galaxies with a DESI spectrum and spans inside its observed range;
4. 1, 4, `max_inputs` and 2 × `max_inputs` concurrent clients, each a thread with its own connection, sending `/search` with 4 patches at 32 matches.

Latency includes Modal's ingress, not the starter's network. Only the checked-out commit is timed.

### Last run

| Run              |                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| Date             | 2026-10-03                                                                                        |
| Commit           | `ff2ee6a`                                                                                         |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                        |
| Server           | 8 CPU, 8 GiB requested, 32 GiB limit; `max_inputs=16`, `max_containers=1`, `scaledown_window=300` |
| Client           | 1 CPU; 17 CPUs visible, faiss at 1 thread; AMD family 25, model 17                                |
| Runs             | 30 warm per figure after one discarded; one cold                                                  |

**`/search`**, warm, 32 matches unless stated:

| Query                  | p50    | p95    |
| ---------------------- | ------ | ------ |
| 4 patches, 8 matches   | 97 ms  | 111 ms |
| 4 patches              | 161 ms | 174 ms |
| 4 patches, 128 matches | 405 ms | 451 ms |
| 4 spans                | 191 ms | 202 ms |
| 4 patches and 4 spans  | 168 ms | 192 ms |

`matches` sets the cost: each match is rescored from 576 reconstructed vectors. Over the 51 ms `/meta` floor, 32 matches add 161 − 51 = 110 ms at p50, and 128 add 405 − 51 = 354 ms.

**Other endpoints**, warm:

| Endpoint                      | p50   | p95   |
| ----------------------------- | ----- | ----- |
| `/meta`                       | 51 ms | 59 ms |
| `/galaxy/{g}/image`           | 53 ms | 62 ms |
| `/galaxy/{g}/image/tokens`    | 59 ms | 69 ms |
| `/galaxy/{g}`                 | 61 ms | 74 ms |
| `/galaxy/{g}/spectrum`        | 57 ms | 67 ms |
| `/galaxy/{g}/spectrum/tokens` | 52 ms | 58 ms |

All six are within 61 − 51 = 10 ms at p50. What the floor is made of is **unmeasured**.

**Cold start.** A request to a fresh container took **19.0 s**. `lifespan` loads `index` and `starts`; the first `/meta` loads `dataset` and `labels`. How the 19.0 s splits between container start and loads is **unmeasured** in this run; `search_performance` times the loads in its own run. With `scaledown_window=300`, a visitor more than five minutes after the last waits the full 19 s; `max_containers=1` leaves no second container to answer.

**Concurrency:**

| Clients | p50    | p95    | Requests/s |
| ------- | ------ | ------ | ---------- |
| 1       | 161 ms | 181 ms | 6.16       |
| 4       | 300 ms | 391 ms | 13.00      |
| 16      | 1.34 s | 1.92 s | 11.62      |
| 32      | 2.56 s | 3.22 s | 12.30      |

**Throughput peaks at 13.00 requests/s with 4 clients** and holds at 11.62–12.30 from 16 up; latency grows with the queue (32 / 12.30 = 2.60 s, against a p50 of 2.56 s). Whether the server or the 1-CPU client sets the ceiling is **unmeasured**.

## `search_quality`

A container on the build image, with the build jobs' CPU, memory and volume, loads every patch and span embedding, then runs `--per-kind` queries of each kind at 32 matches:

- `patches`: 4 patches of a galaxy drawn from all galaxies.
- `paired_patches`, `spans`, `both`: one draw of galaxies with a DESI spectrum, each with 4 patches and 4 spans inside its observed range, queried with the patches, the spans, and both.

A query's recall is the share of `exact_ranking`'s 32 galaxies (a brute force over float32 embeddings) that `search()` returns at the served `PROBE` and `NPROBE`. Each kind reports the mean, the minimum, the share that found all 32, and the share that returned fewer than 32.

### Last run

| Run              |                                                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| Date             | 2026-10-03                                                                                                        |
| Commit           | `5c65e2d-dirty`: uncommitted changes on top of `5c65e2d`                                                          |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                        |
| Machine          | 16 CPU, 32 GiB requested, 128 GiB limit; 32 CPUs visible, faiss and OpenMP at 16 threads; AMD family 25, model 17 |
| Queries          | 100 per kind, `PROBE=2048`, `NPROBE=64`                                                                           |

| Query                 | Drawn from | Mean  | Lowest | All 32 found | Fewer than 32 returned |
| --------------------- | ---------- | ----- | ------ | ------------ | ---------------------- |
| 4 patches             | all        | 98.0% | 65.6%  | 74%          | 0%                     |
| 4 patches             | DESI       | 96.7% | 71.9%  | 63%          | 0%                     |
| 4 spans               | DESI       | 96.7% | 53.1%  | 70%          | 0%                     |
| 4 patches and 4 spans | DESI       | 94.3% | 43.8%  | 47%          | 0%                     |

## Scaling ceilings

| Ceiling          | Now           | Breaks at                                                                                 |
| ---------------- | ------------- | ----------------------------------------------------------------------------------------- |
| Cold start       | 19.0 s        | a proxy or browser timeout; which one, and at what length, is unmeasured                  |
| Serving capacity | one container | 13.00 requests/s peak `/search` throughput at 4 clients; `max_containers=1` is a hard cap |
