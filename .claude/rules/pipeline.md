---
paths:
  - "app/encode.py"
  - "app/parametric_umap.py"
  - "app/dataset.py"
  - "app/search.py"
  - "app/cutouts.py"
  - "app/spectra.py"
  - "modal_app.py"
  - "pyproject.toml"
  - "scripts/alphauniverse_cosmos.py"
---

# Pipeline

- A pull request that changes what a `generate_*` job writes names the jobs that must be rerun, in order, before the change can deploy.
- A change to an artifact's schema updates its schema in `docs/pipeline.md`, its loader's checks and `scripts/fixture.py` together.
