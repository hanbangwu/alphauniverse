# Benchmarks

Every benchmark runs on Modal and measures the production artifacts on the volume.

The stored reports predate #215's rename and use the old names for image tokens (`patches`), spectrum tokens (`spans`) and table values (`scalars`), as in `span_maps`, `paired_patches` and `similarity scalars matches=32`.

| Script                | Measures                                        | Runs on                                   | Report                                     |
| --------------------- | ----------------------------------------------- | ----------------------------------------- | ------------------------------------------ |
| `search_performance`  | loads and `pql.search()` stages, across commits | Modal, a container with the server's spec | `docs/benchmarks/search_performance.json`  |
| `backend_performance` | HTTP latency, cold start and concurrency        | Modal, an ephemeral server and a client   | `docs/benchmarks/backend_performance.json` |
| `projection_quality`  | how the projector keeps neighbours              | Modal, a build-image container            | `docs/benchmarks/projection_quality.json`  |
| `text_search_quality` | text queries against catalogue cuts             | Modal, a build-image container with a GPU | `docs/benchmarks/text_search_quality.json` |
| `pql_quality`         | PQL's rankings on held-out galaxies             | Modal, a build-image container with a GPU | `docs/benchmarks/pql_quality.json`         |
| `compression_quality` | the prediction store's compression schemes      | Modal, a build-image container with a GPU | `docs/benchmarks/compression_quality.json` |

```sh
uv run modal run -m scripts.benchmarks.search_performance   # --runs, default 30
uv run modal run -m scripts.benchmarks.backend_performance  # --runs, default 30
uv run modal run -m scripts.benchmarks.projection_quality
uv run modal run -m scripts.benchmarks.text_search_quality
uv run modal run -m scripts.benchmarks.pql_quality          # --sample, default 1000
uv run modal run -m scripts.benchmarks.compression_quality  # --sample 1024, --fit 512, --queries 200
```

- The performance scripts refuse to run with uncommitted changes.
- A pull request posts its run's report on the pull request and commits none; after a round, a docs pull request reruns the scripts on `main` and commits their reports.
- Figures compare only within one run: not across runs, machines or thread layouts.
- **Unmeasured** marks a figure read off the code.
- A report records `memory`, read after the measured work: the process's peak resident set (`peak_rss_mib`, from `getrusage`), which counts the pages of memory-mapped files it read, such as `predictions`, and the most pyarrow's memory pool held at once (`arrow_peak_mib`). `resident_files_mib` and `resident_anonymous_mib` split the resident memory at the end of the run, from `/proc/self/smaps`, into pages of memory-mapped files and the rest. Under Modal's gVisor a mapped file that is touched stays resident almost whole (the 16.5 GiB index and 23 GiB of the Hugging Face dataset in Benchmark run 38022404256), so, in a process that did not inherit its parent's peak and dropped no file pages, `peak_rss_mib` minus `resident_files_mib` bounds the peak of the rest from below. Only anonymous memory counts against a Modal container's memory limit, so `resident_anonymous_mib`, not `peak_rss_mib`, is the figure to compare with it: in Benchmark run 38029573811, a container limited to 4 GiB read 40.36 GiB of mapped artifacts and finished with 22,133 MiB of their pages resident, and one allocating anonymous memory was killed at 3,677 MiB on all 8 attempts. None of these counts GPU memory. A figure covers everything the benchmark process held, including data the server never builds. In `search_performance` each round's subprocess records its own, if its version's code does, and so does the parent; Linux carries the parent's peak at that moment into a child at `exec`, so a round near the parent's figure may be reading the parent's. `backend_performance` records none, since its client cannot read the server's memory.

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
2. `lifespan`'s search loads: `galaxy_count`, `labels`, `tokens`, `predictions`, `basis`;
3. the stages of `pql.search()`, first query and warm, at 4 image tokens and 32 matches: `forms` (the query galaxy's forms), `scan` (every galaxy's mode sums), `combine`, `similarity`, `order`, `maps` and `predicted`;
4. `pql.search()` whole at 8, 32 and 128 matches;
5. under `kinds`, outside `total_p50_ms`: first query and warm, the scan (`scores` and the ranking) and `maps` for the top 32, at 4 image tokens, 16 contiguous spectrum tokens, all 272 spectrum tokens and 4 Legacy Survey table values.

The loads and the warm stages also record wall, user and system milliseconds (`usage_ms`), the warm stages as means. Modal runs containers under gVisor, which samples CPU time in 10 ms ticks and reports no page faults, so a CPU figure is coarse unless it spans many ticks. CPU is the whole process's, so OpenMP and OpenBLAS workers that spin after one stage's parallel region are charged to the next. `thread_pools` lists every BLAS and OpenMP pool in the process with its thread count.

Versions are the checked-out commit (after), `origin/main` (before), and the stored report's best unless its code matches one of those. Each is a `git archive` of `app/`, `scripts/` and `modal_app.py`, run in a subprocess with after's locked dependencies, in mirrored order: after, before, best, then back. After's first round has the cold page cache. A failed round records its error.

The best version has the lowest `total_p50_ms` (sum of warm stage medians at 32 matches) averaged over its rounds, among versions whose rounds all succeeded. A stored best missing from the clone is dropped with a note. If no version succeeds, best keeps the stored commit without its figure.

The stored report (2026-10-04, `6aa4197`) timed the cosine search, which is gone. The PQL search is unmeasured.

## `backend_performance`

`modal run` starts an ephemeral `fastapi_app` from the checked-out source, with its image, CPU, memory and concurrency. A 1-CPU client in a separate container times, in order:

1. one cold `/meta`, one cold `/search`, then one `/search/text`, the first to load EmbeddingGemma and `aion_gemma_space`;
2. warm, `runs` times each after one warm-up: `/meta`, image, image tokens, galaxy, table, `/search/text` cycling through `text_search_quality`'s six queries, `/search` with 4 Legacy Survey table values at 32 matches, and `/search` with 4 image tokens at 8, 32 and 128 matches;
3. warm: both spectrum routes, and `/search` at 32 matches with 4 spectrum tokens, alone and with 4 image tokens, on galaxies with a DESI spectrum and spectrum tokens inside its observed range;
4. 1, 4, `max_inputs` and 2 × `max_inputs` concurrent clients, each a thread with its own connection, sending `/search` with 4 image tokens at 32 matches.

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

| Query                                | p50    | p95    |
| ------------------------------------ | ------ | ------ |
| 4 image tokens, 8 matches            | 164 ms | 174 ms |
| 4 image tokens                       | 225 ms | 255 ms |
| 4 image tokens, 128 matches          | 429 ms | 503 ms |
| 4 spectrum tokens                    | 253 ms | 276 ms |
| 4 image tokens and 4 spectrum tokens | 237 ms | 258 ms |
| 4 Legacy Survey table values         | 231 ms | 264 ms |

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

## `projection_quality`

A container on the build image, with 16 CPU, 16 GiB requested, a 64 GiB limit and no GPU, redraws the projector's sample and its validation split, reads `SIZE` (10,000) of the validation rows from `encoded`, and projects them with the stored `parametric_umap`. At 15 and 100 neighbours it reports:

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

## `pql_quality`

A container on the build image, with an L4 GPU, 16 CPU, 32 GiB requested and a 128 GiB limit, draws `--sample` galaxies with a spectrum and a DESI or SDSS redshift. It predicts each one again with its spectra and its redshift token removed, with the job's own code. Per galaxy it queries 16 contiguous observed spectrum tokens of its first spectrum survey, all its observed spectrum tokens, 4 Legacy Survey table values, and 4 HSC table values where it has an HSC match. Each query is scored by `app.pql` against every galaxy, and reports the mean with a 95% bootstrap interval over queries:

- `redshift`: the median |Δz|/(1+z) of the top 10 galaxies with a redshift, the query galaxy excluded.
- `identity` (spectrum token queries): whether the query galaxy, spectrum and redshift removed, ranks in the top 10 among every other galaxy.
- `availability` and `evidence` (spectrum token queries): over 20 coin flips that show each other sampled galaxy with or without its spectrum, the share with a spectrum among the top 32 minus its share overall, for galaxies at another redshift (|Δz|/(1+z) ≥ 0.01) and at the same one.

It needs `predictions` and `prediction_basis` from `generate_predictions`. Unmeasured.

## `compression_quality`

Measures how each way of storing a predicted distribution changes PQL's scores and rankings, for every mode in `app.predictions.MODES`. Unlike the other benchmarks it predicts its own galaxies, densely, from `tokens`; it also reads `mean_points` for the galaxy count and the dataset for spectra's wavelengths and redshifts. Each mode runs in its own container, as `pql_quality`'s, with a 24-hour timeout, and the local entrypoint combines them.

- **Galaxies** (`SEED`): `--sample` scored galaxies, up to half of them drawn from galaxies with SDSS, then from galaxies with HSC or DESI up to half the sample, the rest from all galaxies; and `--fit` other galaxies whose predictions fit the mode's PCA basis, so every scored galaxy is out of sample. Image modes are held in fp16, the rest in fp32.
- **Schemes** (`scripts/benchmarks/compression.py`, 218 per mode): the store's current scheme, with the fit galaxies' basis for spectra; dense; the top k codes; the fewest codes holding mass p; and PCA coefficients. Top-k and top-p either spread the rest of the mass evenly or drop it, and every scheme is stored at fp32, fp16, bf16, 8 bits or 4 bits. The levels are listed in #214.
- **Queries**: `--queries` galaxies per mode, drawn from those observing it, each with selections of one slot, a block or window, and all slots, as in `selections`. Each query is scored against the sample and against a copy of itself predicted with the mode's tokens hidden. For table values and the redshift, the exact reference, the current scheme and the dense log-probability schemes take the overlap as a log-sum-exp of the log-probabilities, as `app.pql` does; every other overlap is floored at float32's smallest normal number, and at `app.pql`'s spectrum token floor for PCA schemes.
- **Per scheme**: bytes per galaxy; the share of floored overlaps; the median and 99th percentile of |log overlap − exact|; one query's scan of 16 slots on the CPU; and per selection size, with the gallery compressed alone and with both sides compressed, the Spearman correlation of the summed score with the dense one (tied ranks averaged), top-10 and top-32 overlap, identity (the hidden copy ranks first) and the median top-10 |Δz|/(1+z).
- **Choices**: per mode, the smallest scheme whose both-sides figures meet `TARGETS` at every size (identity may not drop by more than its target); and for corpus totals of 1, 2, 4 and 8 GB, the schemes that maximise the lowest median Spearman over modes and sizes.
- **Check**: for the image modes, the log overlaps of 64 galaxies with fp16 and fp32 references.

The report is too large for the workflow's step summary; it is in the run's artifact. Unmeasured.

## Scaling ceilings

| Ceiling          | Now           | Breaks at                                                                                             |
| ---------------- | ------------- | ----------------------------------------------------------------------------------------------------- |
| Cold start       | 18.1 s        | a proxy or browser timeout; which one, and at what length, is unmeasured                              |
| Serving capacity | one container | 15.63 requests/s `/search` throughput at 32 clients, the most timed; `max_containers=1` is a hard cap |
