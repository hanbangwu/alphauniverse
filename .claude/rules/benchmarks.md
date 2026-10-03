---
paths:
  - "scripts/benchmark.py"
  - "scripts/recall.py"
  - "docs/performance.md"
  - "modal_app.py"
  - "docs/benchmarks/**"
  - ".github/workflows/benchmark.yml"
---

# Benchmarks

- A benchmark measures only the production artifacts (on Modal).
- If the code under test needs artifacts the volume does not hold yet, name the `generate_*` jobs that build them, in order, and wait for a maintainer to ask for that build: they write to the volume the deployed app serves from.
- A run times the `search()` stages of the checked-out commit, the head of `main`, and the stored report's best version when it is neither, in one container, so they compare on the same host. Compare versions only within one report, and loads only round by round, since they depend on which version loaded the artifacts first. Request latency is timed for the checked-out commit only.
- A pull request that records a run may compare its versions in the description, from that run's report.
- Each version's subprocess calls that version's `stages(runs)` through Modal's `.local()`, so its name, its first argument and the `total_p50_ms` it returns stay as they are.
- A change expected to affect performance lists a benchmark run among the goals in its issue, and agreeing to the goals is a maintainer asking for that run. Otherwise a benchmark or recall run happens only when a maintainer asks. It is started by hand, locally or from the Benchmark workflow, never by a push, a pull request or a schedule.
- The pattern follows Modal's benchmarking tool, [`stopwatch`](https://github.com/modal-labs/stopwatch): `modal run` starts an ephemeral copy of the serving function from the checked-out source, and a client in a separate container sends it requests. The server takes its image, CPU, memory and concurrency from `fastapi_app`'s definition, so the two cannot drift.
- Latency is measured by that client, so it includes Modal's ingress but not the network of whoever started the run. Stage splits inside `search()` are timed in a container with the same spec as the server.
- Cold and warm are separate measurements. Cold is a request to a freshly started container; warm discards one warm-up request, then reports p50, p95 and the run count.
- A run's report records the commits, the dataset revision, the server's CPU, memory and concurrency settings from the constants `fastapi_app` is defined with, the client's Modal spec, the thread configuration, the CPU identity of each container it ran in, and the date.
- Every latency figure in `docs/performance.md` comes from one run of `scripts/benchmark.py`, and the recall figure from one run of `scripts/recall.py`; the doc names each run. The next latency update takes its figures from the after version in `docs/benchmarks/latest.json`, and the next recall update from `docs/benchmarks/recall.json`. An update replaces every figure the new run covers, and removes or marks unmeasured every figure it does not, rather than keep one from an older run.
