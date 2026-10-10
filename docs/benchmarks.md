# Benchmarks

Every benchmark runs on Modal and measures the production artifacts on the volume.

| Script                | Measures                                     | Runs on                                   | Report                                     |
| --------------------- | -------------------------------------------- | ----------------------------------------- | ------------------------------------------ |
| `search_performance`  | loads and `search()` stages, across commits  | Modal, a container with the server's spec | `docs/benchmarks/search_performance.json`  |
| `backend_performance` | HTTP latency, cold start and concurrency     | Modal, an ephemeral server and a client   | `docs/benchmarks/backend_performance.json` |
| `search_quality`      | recall of `search()` against an exact search | Modal, a build-image container            | `docs/benchmarks/search_quality.json`      |
| `projection_quality`  | how the projector keeps neighbours           | Modal, a build-image container            | `docs/benchmarks/projection_quality.json`  |
| `text_search_quality` | text queries against catalogue cuts          | Modal, a build-image container with a GPU | `docs/benchmarks/text_search_quality.json` |

```sh
uv run modal run -m scripts.benchmarks.search_performance   # --runs, default 30
uv run modal run -m scripts.benchmarks.backend_performance  # --runs, default 30
uv run modal run -m scripts.benchmarks.search_quality       # --per-kind, default 100
uv run modal run -m scripts.benchmarks.projection_quality
uv run modal run -m scripts.benchmarks.text_search_quality
```

- The performance scripts refuse to run with uncommitted changes.
- A pull request posts its run's report on the pull request and commits none; after a round, a docs pull request reruns the scripts on `main` and commits their reports.
- Figures compare only within one run: not across runs, machines or thread layouts.
- **Unmeasured** marks a figure read off the code.

## From GitHub Actions

Start the Benchmark workflow by hand from the Actions tab. Pick a branch that contains the workflow, a script, and optionally `args`, the script's flags as listed above (empty keeps the defaults). The job uses the deploy job's Modal token. The report appears in the run's summary and as its artifact; the job does not commit it. A job past six hours stops and leaves no report.

## Cost model

What a first visitor to an idle site waits for, from the last `backend_performance` run:

| Step                             | Cost                        |
| -------------------------------- | --------------------------- |
| Container start + loads, `/meta` | **18.1 s**                  |
| First `/search` after that       | 0.52 s                      |
| First `/search/text` after that  | 35.5 s                      |
| Everything warm after that       | 0.13–0.43 s p50 per request |

**A cold visit shows a spinner until `/meta` returns**, 18.1 s after the browser sends it. Cold start dominates, then the first text search.

## `search_performance`

A container with the server's spec runs, per version:

1. `import app.main`, in a fresh subprocess. A version's first import also compiles its `app/`, so compare later rounds;
2. `read_index` and `make_direct_map`, the two calls `index()` loads with, timed on a separate copy, so in the first round they take the index's cold reads;
3. `lifespan`'s loads: `galaxy_count`, `labels`, `index`, `tokens` (which `starts` loads), `starts`. `index` finds the pages step 2 read, so its cold cost is step 2's;
4. the stages of `search()`, first query and warm, at 4 patches and 32 matches;
5. `search()` whole at 8, 32 and 128 matches.

The split, the loads and the warm stages also record wall, user and system milliseconds (`usage_ms`), the warm stages as means. Modal runs containers under gVisor, which samples CPU time in 10 ms ticks and reports no page faults, so a CPU figure is coarse unless it spans many ticks. Cold reads are charged to user time: in the last run's first round, `make_direct_map` took 8,465 ms of wall time and 8,310 ms of user CPU, none of system. CPU is the whole process's, so OpenMP and OpenBLAS workers that spin after one stage's parallel region are charged to the next. `thread_pools` lists every BLAS and OpenMP pool in the process with its thread count. An OpenMP count is per calling thread: `faiss.omp_set_num_threads` changes only its caller, so a server's thread layout is set through the environment.

Versions are the checked-out commit (after), `origin/main` (before), and the stored report's best unless its code matches one of those. Each is a `git archive` of `app/`, `scripts/` and `modal_app.py`, run in a subprocess with after's locked dependencies, in mirrored order: after, before, best, then back. After's first round has the cold page cache. A failed round records its error.

The best version has the lowest `total_p50_ms` (sum of warm stage medians at 32 matches) averaged over its rounds, among versions whose rounds all succeeded. A stored best missing from the clone is dropped with a note. If no version succeeds, best keeps the stored commit without its figure.

### Last run

| Run              |                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------ |
| Date             | 2026-10-04                                                                                                   |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                   |
| Versions         | `6aa4197` as after and as before (`main`); the stored best, `ff2ee6a`, is not in the clone and was dropped   |
| Container        | 8 CPU, 8 GiB requested, 32 GiB limit; 24 CPUs visible, faiss and OpenMP at 8 threads; AMD family 25, model 1 |
| Rounds           | 4, two per version; 30 warm queries per figure after one discarded                                           |

Figures are after's. Warm figures average its two rounds. After and before are the same commit, so later rounds range over all three.

**Loads.** Later rounds read files already in the page cache:

| Load              | First round | Later rounds  |
| ----------------- | ----------- | ------------- |
| `import app.main` | 5.19 s      | 2.38–3.58 s   |
| `read_index`      | 0.106 s     | 0.016–0.018 s |
| `make_direct_map` | 8.47 s      | 0.16–0.18 s   |
| `galaxy_count`    | 9.0 ms      | 2.4–2.9 ms    |
| `labels`          | 16.5 ms     | 5.7–6.6 ms    |
| `index`           | 323 ms      | 175–192 ms    |
| `tokens`          | 56.3 ms     | 38.2–60.2 ms  |
| `starts`          | 4.6 ms      | 3.8–5.8 ms    |

`read_index` and `make_direct_map` are step 2's split, which takes the cold reads; `index` then finds those pages. `faiss.read_index` memory-maps the index, so pages fault in as queries touch them; `make_direct_map()` reads every list's ids.

**Stages**, 4 patches, 32 matches. The run predates `scalar_maps`, which the script now times:

| Stage         | First query | Warm p50    | Warm share |
| ------------- | ----------- | ----------- | ---------- |
| `centroid`    | 0.72 ms     | 0.31 ms     | 0.4 %      |
| `candidates`  | 21.8 ms     | 5.03 ms     | 5.8 %      |
| **`vectors`** | 88.0 ms     | **79.8 ms** | **91.5 %** |
| `score_maps`  | 7.59 ms     | 1.51 ms     | 1.7 %      |
| `span_maps`   | 12.5 ms     | 0.37 ms     | 0.4 %      |
| `scalar_maps` | unmeasured  | unmeasured  |            |
| `rank`        | 0.35 ms     | 0.14 ms     | 0.2 %      |
| Total         | 131.0 ms    | 87.2 ms     |            |

- **`vectors` dominates**: it reconstructs 33 × 576 = 19,008 patch vectors in 79.8 ms, 4.2 µs each. `reconstruct_batch` walks the IVF direct map one vector at a time, not a contiguous read.
- `candidates` and `vectors` are faiss-parallel and `score_maps` contends with their threads, so stage figures compare only at the same thread configuration.

**Whole `search()`**, 4 patches, warm p50:

| Matches | p50      |
| ------- | -------- |
| 8       | 29.1 ms  |
| 32      | 89.5 ms  |
| 128     | 306.3 ms |

**Rounds.** Warm totals are 92.288 and 82.014 ms (after) and 94.215 and 92.371 ms (before): one commit's rounds differ by up to 94.215 − 82.014 = 12.201 ms.

## `backend_performance`

`modal run` starts an ephemeral `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency. A 1-CPU client in a separate container times, in order:

1. one cold `/meta`, one cold `/search`, then one `/search/text`, the first to load EmbeddingGemma and `aion_gemma_space`;
2. warm, `runs` times each after one warm-up: `/meta`, image, image tokens, galaxy, table, `/search/text` cycling through `text_search_quality`'s six queries, `/search` with 4 Legacy Survey scalars at 32 matches, and `/search` with 4 patches at 8, 32 and 128 matches;
3. warm: both spectrum routes, and `/search` at 32 matches with 4 spans, alone and with 4 patches, on galaxies with a DESI spectrum and spans inside its observed range;
4. 1, 4, `max_inputs` and 2 × `max_inputs` concurrent clients, each a thread with its own connection, sending `/search` with 4 patches at 32 matches.

Latency includes Modal's ingress, not the starter's network. Only the checked-out commit is timed.

### Last run

| Run              |                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------- |
| Date             | 2026-10-10                                                                                        |
| Commit           | `853bac8`, #190 and #191 on `37bff8a`                                                             |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                        |
| Server           | 8 CPU, 8 GiB requested, 32 GiB limit; `max_inputs=16`, `max_containers=1`, `scaledown_window=300` |
| Client           | 1 CPU; 17 CPUs visible, faiss at 1 thread; AMD family 25, model 1                                 |
| Runs             | 30 warm per figure after one discarded; one cold                                                  |

**`/search`**, warm, 32 matches unless stated:

| Query                   | p50    | p95    |
| ----------------------- | ------ | ------ |
| 4 patches, 8 matches    | 164 ms | 174 ms |
| 4 patches               | 225 ms | 255 ms |
| 4 patches, 128 matches  | 429 ms | 503 ms |
| 4 spans                 | 253 ms | 276 ms |
| 4 patches and 4 spans   | 237 ms | 258 ms |
| 4 Legacy Survey scalars | 231 ms | 264 ms |

`matches` sets the cost. Over the 128 ms `/meta` floor, 32 matches add 225 − 128 = 97 ms at p50, and 128 add 429 − 128 = 301 ms.

**Other endpoints**, warm:

| Endpoint                      | p50    | p95    |
| ----------------------------- | ------ | ------ |
| `/meta`                       | 128 ms | 135 ms |
| `/galaxy/{g}/image`           | 140 ms | 151 ms |
| `/galaxy/{g}/image/tokens`    | 126 ms | 133 ms |
| `/galaxy/{g}`                 | 126 ms | 131 ms |
| `/galaxy/{g}/spectrum`        | 130 ms | 145 ms |
| `/galaxy/{g}/spectrum/tokens` | 126 ms | 134 ms |
| `/galaxy/{g}/table`           | 139 ms | 151 ms |
| `/search/text`                | 284 ms | 316 ms |

All but `/search/text` are within 140 − 126 = 14 ms at p50; `/search/text` adds 284 − 128 = 156 ms over the floor. What the floor is made of is **unmeasured**.

**Cold start.** A request to a fresh container took **18.1 s**. How the 18.1 s splits between container start and loads is **unmeasured** in this run; `search_performance` times the loads in its own run. With `scaledown_window=300`, a visitor more than five minutes after the last waits the full 18.1 s; `max_containers=1` leaves no second container to answer. The first `/search/text` after the cold `/search` took **35.5 s**: it loads EmbeddingGemma and `aion_gemma_space` and imports sentence-transformers.

**Concurrency:**

| Clients | p50    | p95    | Requests/s |
| ------- | ------ | ------ | ---------- |
| 1       | 232 ms | 262 ms | 4.26       |
| 4       | 289 ms | 345 ms | 13.42      |
| 16      | 1.05 s | 1.39 s | 15.61      |
| 32      | 2.10 s | 2.64 s | 15.63      |

**Throughput reaches 15.63 requests/s with 32 clients**, the most clients timed, against 15.61 with 16; latency grows with the queue (32 / 15.63 = 2.05 s, against a p50 of 2.10 s). Whether the server or the 1-CPU client sets the ceiling is **unmeasured**.

## `search_quality`

A container on the build image, with 16 CPU, 16 GiB requested, a 64 GiB limit and the volume, runs `--per-kind` queries of each kind at 32 matches:

- `patches`: 4 patches of a galaxy drawn from all galaxies.
- `paired_patches`, `spans`, `both`: one draw of galaxies with a DESI spectrum, each with 4 patches and 4 spans inside its observed range, queried with the patches, the spans, and both.

A query's recall is the share of its exact 32 galaxies that `search()` returns at the served `PROBE` and `NPROBE`. `exact_rankings` sets each query's direction from its galaxy's rows, then brute-forces the float32 embeddings in one streamed pass over `encoded`, `BATCH` galaxies at a time, keeping each galaxy's best score per query. Each kind reports the mean, the minimum, the share that found all 32, the share that searched the index more than once, and the most searches one query took.

### Last run

| Run              |                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| Date             | 2026-10-04                                                                                                       |
| Commit           | `6aa4197`                                                                                                        |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                       |
| Machine          | 16 CPU, 32 GiB requested, 128 GiB limit; 32 CPUs visible, faiss and OpenMP at 16 threads; AMD family 25, model 1 |
| Queries          | 100 per kind, `PROBE=2048`, `NPROBE=64`                                                                          |

| Query                 | Drawn from | Mean  | Lowest | All 32 found | Searched again | Most searches |
| --------------------- | ---------- | ----- | ------ | ------------ | -------------- | ------------- |
| 4 patches             | all        | 98.0% | 65.6%  | 74%          | 0%             | 1             |
| 4 patches             | DESI       | 96.7% | 71.9%  | 63%          | 0%             | 1             |
| 4 spans               | DESI       | 96.7% | 53.1%  | 70%          | 0%             | 1             |
| 4 patches and 4 spans | DESI       | 94.3% | 43.8%  | 47%          | 0%             | 1             |

## `projection_quality`

A container on the build image, with 16 CPU, 16 GiB requested, a 64 GiB limit and no GPU, redraws the projector's sample and its validation split, takes `SIZE` (10,000) validation rows, and projects them with the stored `parametric_umap`. At 15 and 100 neighbours it reports:

- `preservation`: the mean share of a row's nearest neighbours by cosine distance in 768-d that are also its nearest in 2-d.
- `trustworthiness`: scikit-learn's `trustworthiness`, cosine in 768-d, which penalises 2-d neighbours that are far apart in 768-d.

The rows are held out only if `parametric_umap` was trained with the validation split from the same `encoded`; a projector from before the split trained on them. `trustworthiness` holds all pairwise distances, so its memory grows with the square of `SIZE`.

### Last run

| Run              |                                                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Date             | 2026-10-05                                                                                                                |
| Commit           | `3005bac-dirty`: uncommitted changes on top of `3005bac`                                                                  |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                                |
| Machine          | 16 CPU, 32 GiB requested, 128 GiB limit, no GPU; 32 CPUs visible, faiss and OpenMP at 16 threads; AMD family 25, model 17 |
| Projector        | trained with `SAMPLE=5000000`, `EPOCHS=20`, `VALIDATION=0.3`                                                              |
| Rows             | 10,000 validation rows                                                                                                    |

| Neighbours | Preservation | Trustworthiness |
| ---------- | ------------ | --------------- |
| 15         | 0.3064       | 0.8926          |
| 100        | 0.3336       | 0.8173          |

At 15 neighbours, 0.3064 × 15 = 4.6 of a row's 15 nearest neighbours in 768-d are among its 15 nearest in 2-d. Random 2-d positions would keep 15 / 9,999 = 0.0015 of them, and score a trustworthiness near 0.5.

## `text_search_quality`

A container on the build image, with an L4 GPU, 16 CPU, 16 GiB requested and a 64 GiB limit, takes the `pairs` rows of the galaxies `generate_alignment` held out, and embeds each query in `CUTS` with EmbeddingGemma. Each query's answer is a cut on a PROVABGS property: stellar mass, specific star formation rate or redshift; its population is the galaxies with that property. It ranks the population in three spaces: the EmbeddingGemma embeddings, and the AION embeddings through the linear and the MLP maps in `alignment`. Each space is ranked raw and centred: centring subtracts the documents' mean from the documents and the queries' mean from the queries, then renormalises. Per query, space and centring it reports the base rate, scikit-learn's `average_precision_score` and the precision at 10 and 100.

The dataset holds 7 galaxies with a GZ10 label, too few to score morphology queries.

### Last run

| Run              |                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Date             | 2026-10-07                                                                                                               |
| Commit           | `16730a0-dirty`: uncommitted changes on top of `16730a0`                                                                 |
| Dataset revision | `e250e43c35e63523ee3940c9543302c29ca56437`                                                                               |
| Machine          | 16 CPU, 32 GiB requested, 128 GiB limit, one L4; 32 CPUs visible, faiss and OpenMP at 16 threads; AMD family 25, model 1 |
| Galaxies         | 5,211 held out by `generate_alignment`                                                                                   |

Average precision, raw:

| Query                                                   | Base rate | EmbeddingGemma | Linear | MLP    |
| ------------------------------------------------------- | --------- | -------------- | ------ | ------ |
| A massive galaxy, stellar mass above 10^11 solar masses | 0.3604    | 0.5654         | 0.5902 | 0.5929 |
| A low-mass dwarf galaxy                                 | 0.0558    | 0.0413         | 0.0437 | 0.0468 |
| A star-forming galaxy                                   | 0.1066    | 0.0933         | 0.1020 | 0.0973 |
| A quiescent galaxy with no ongoing star formation       | 0.3236    | 0.3028         | 0.2668 | 0.2525 |
| A distant galaxy at redshift above 0.5                  | 0.0178    | 0.0153         | 0.0140 | 0.0152 |
| A nearby galaxy at redshift below 0.1                   | 0.0876    | 0.1096         | 0.1088 | 0.1283 |

Raw, only the stellar-mass query beats its base rate by more than 0.05 in any space. Centred figures and precision at 10 and 100 are in `docs/benchmarks/text_search_quality.json`.

## Scaling ceilings

| Ceiling          | Now           | Breaks at                                                                                             |
| ---------------- | ------------- | ----------------------------------------------------------------------------------------------------- |
| Cold start       | 18.1 s        | a proxy or browser timeout; which one, and at what length, is unmeasured                              |
| Serving capacity | one container | 15.63 requests/s `/search` throughput at 32 clients, the most timed; `max_containers=1` is a hard cap |
