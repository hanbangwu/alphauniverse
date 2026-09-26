---
paths:
  - "scripts/benchmark.py"
  - "docs/performance.md"
  - "modal_app.py"
  - "docs/benchmarks/**"
---

# Benchmarks

- A benchmark measures only the production artifacts (on Modal).
- If the code under test needs artifacts the volume does not hold yet, name the `generate_*` jobs that build them, in order, and wait for a maintainer to ask for that build: they write to the volume the deployed app serves from.
- A run times the `search()` stages of the change and of the head of `main` in one container, so they compare on the same host. It also times the best version the stored report names, when that is neither. Compare versions only within one report. Startup loads depend on which version loaded the artifacts first, so compare them round by round. Request latency is timed for the change only.
- A pull request that records a run may compare its versions in the description, from that run's report.
- Each version's subprocess calls that version's `stages(runs)` through Modal's `.local()`, so its name, its first argument and the `total_p50_ms` it returns stay as they are.
- A change expected to affect performance lists a benchmark run among the goals in its issue, and agreeing to the goals is a maintainer asking for that run. Otherwise a benchmark runs only when a maintainer asks. Never in CI or on a schedule.
- The pattern follows Modal's own benchmarking tool, [`stopwatch`](https://github.com/modal-labs/stopwatch): `modal run` starts an ephemeral copy of the serving function from the checked-out source, and a client in a separate Modal container sends it requests. The server takes its image, CPU, memory and concurrency from `fastapi_app`'s own definition, so the two cannot drift.
- Latency is measured by that client, so it includes Modal's ingress but not the network of whoever started the run. Stage splits inside `search()` are timed in a container with the same spec as the server.
- Cold and warm are separate measurements. Cold is a request to a freshly started container; warm discards one warm-up request, then reports p50, p95 and the run count.
- A run's report records the commit, the dataset revision, the Modal spec of server and client, the thread configuration, the CPU model of each container it ran in, and the date.
- Every figure in `docs/performance.md` comes from one run, and the doc names it. The next update takes its figures from the after version in `docs/benchmarks/latest.json`. Updating the doc replaces every figure the new run covers. A figure the new run does not cover is removed or marked unmeasured, not kept from an older run.
