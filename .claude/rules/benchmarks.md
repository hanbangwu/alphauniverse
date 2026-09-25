---
paths:
  - "scripts/benchmark.py"
  - "docs/performance.md"
  - "modal_app.py"
---

# Benchmarks

- A benchmark measures only the production artifacts (on Modal).
- Always measure and never estimate figures.
- If the code under test needs artifacts the volume does not hold yet, name the `generate_*` jobs that build them, in order, and wait for a maintainer to ask for that build: they write to the volume the deployed app serves from.
- A benchmark measures the code as it is now. Do not add "before/after". No one cares about the "before".
- Benchmarks run only when the user asks, directly or through Claude. Never in CI or on a schedule.
- The pattern follows Modal's own benchmarking tool, [`stopwatch`](https://github.com/modal-labs/stopwatch): `modal run` starts an ephemeral copy of the serving function from the checked-out source, and a client in a separate Modal container sends it requests. The server takes its image, CPU, memory and concurrency from `fastapi_app`'s own definition, so the two cannot drift.
- Latency is measured by that client, so it includes Modal's ingress but not the network of whoever started the run. Stage splits inside `search()` are timed in a container with the same spec as the server.
- Cold and warm are separate measurements. Cold is a request to a freshly started container; warm discards one warm-up request, then reports p50, p95 and the run count.
- A run's report records the commit, the dataset revision, the Modal spec of server and client, the thread configuration and the date.
- Every figure in `docs/performance.md` comes from one run, and the doc names it. Updating the doc replaces every figure the new run covers. A figure the new run does not cover is removed or marked unmeasured, not kept from an older run.
