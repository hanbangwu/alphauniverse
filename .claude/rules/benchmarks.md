---
paths:
  - "scripts/benchmarks/**"
  - "docs/benchmarks.rst"
  - "modal_app.py"
  - "docs/benchmarks/**"
  - ".github/workflows/benchmark.yml"
---

# Benchmarks

- Every benchmark runs on Modal, never locally, and measures only the production artifacts on the volume.
- Each benchmark is a standing benchmark or a study, and its issue and its section in `docs/benchmarks.rst` say which. A standing benchmark measures what ships, in its shipped configuration only, and reruns when the code it measures changes; alternatives, sweeps and design matrices belong in studies. A study answers one design question; its report records the run that answered it, and it reruns only when that question reopens.
- If the code under test needs artifacts the volume does not hold yet, name the `generate_*` jobs that build them, in order, and wait for a maintainer to ask for that build: they write to the volume the deployed app serves from.
- A run times the `search()` stages of the checked-out commit, and the stored report's best version when its code differs, in one container, so they compare on the same host. Compare versions only within one report, and loads only round by round, since they depend on which version loaded the artifacts first. Request latency is timed for the checked-out commit only.
- A pull request whose goals list a run posts it on the pull request: the workflow run's link and the comparison of its versions in the description, the report in a collapsed comment. It commits no report. After a round of such pull requests, one docs pull request reruns the standing benchmarks on `main` and replaces their reports.
- Each version's subprocess calls that version's `stages(runs)` through Modal's `.local()`, so its name, its first argument and the `total_p50_ms` it returns stay as they are.
- A change expected to affect performance lists a benchmark run among the goals in its issue, and agreeing to the goals is a maintainer asking for that run. Otherwise a benchmark run happens only when a maintainer asks. It is started by hand, with `modal run` or from the Benchmark workflow, never by a push, a pull request or a schedule.
- The pattern follows Modal's benchmarking tool, [`stopwatch`](https://github.com/modal-labs/stopwatch): `modal run` starts an ephemeral copy of the serving function from the checked-out source, and a client in a separate container sends it requests. The server takes its image, CPU, memory and concurrency from `fastapi_app`'s definition, so the two cannot drift.
- Latency is measured by that client, so it includes Modal's ingress but not the network of whoever started the run. Stage splits inside `search()` are timed in a container with the same spec as the server.
- Cold and warm are separate measurements. Cold is a request to a freshly started container; warm discards its warm-up requests, then reports p50, p95 and the run count.
- A run's report records the commits, the dataset revision, the server's CPU, memory and concurrency settings from the constants `fastapi_app` is defined with, the thread configuration, the CPU identity of each container it ran in, the date, and the peak memory and end-of-run resident split of every process that runs the measured work, except `backend_performance`'s server, which its client cannot read.
- Compare memory with a container's limit through `resident_anonymous_mib`: pages of memory-mapped files do not count against Modal's limit.
- Code that reads an artifact larger than one batch reads it in batches into preallocated arrays, and never holds a second copy of it.
- `docs/benchmarks.rst` draws every figure from the reports in `docs/benchmarks/` when the docs build, and names each report's commit and date; it never types a figure. It does not show figures from code that no longer exists.
- Committing a report is the only step that updates the page: each script's plot, a function in `docs/benchmark_plots.py` that `docs/benchmarks.rst` calls, reads only its report's keys and shows whatever the report holds. A change to a script that renames or drops a key its plot reads updates the plot in the same pull request, and the docs build in CI checks it against the committed report.
