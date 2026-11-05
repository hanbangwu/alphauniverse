---
paths:
  - "pipeline/**"
  - "core/**"
  - "pyproject.toml"
  - "scripts/alphauniverse_cosmos.py"
---

# Pipeline

- A pull request that changes what a `generate_*` job writes names the jobs that must be rerun, in order, before the change can deploy.
- A change to an artifact's schema updates `scripts/fixture.py` and the artifact's prose in `docs/dev/reference.rst` together.
- Code that reads an artifact larger than one batch reads it in batches into preallocated arrays, and never holds a second copy of it.
- Constants used by more than one `pipeline/` module live in `pipeline/config.py`, and artifact schemas in `pipeline/schemas.py`.
