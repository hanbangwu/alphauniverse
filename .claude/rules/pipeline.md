---
paths:
  - "src/alphauniverse/dataset/dataset.py"
  - "src/alphauniverse/encoding/store.py"
  - "src/alphauniverse/encoding/build/encode.py"
  - "src/alphauniverse/predictions/store.py"
  - "src/alphauniverse/predictions/build/predictions.py"
  - "src/alphauniverse/predictions/build/aion_predictions.py"
  - "src/alphauniverse/predictions/build/observations.py"
  - "src/alphauniverse/predictions/build/grams.py"
  - "src/alphauniverse/predictions/build/embeddings.py"
  - "benchmarks/library/predictions/build/observed_grams.py"
  - "src/alphauniverse/text_search/text_search.py"
  - "src/alphauniverse/text_search/build/alignment.py"
  - "src/alphauniverse/text_search/build/text_views.py"
  - "src/alphauniverse/text_search/build/text_regression.py"
  - "server/projection/build/parametric_umap.py"
  - "server/projection/build/properties.py"
  - "modal_app.py"
  - "pyproject.toml"
  - "scripts/alphauniverse_cosmos.py"
---

# Pipeline

- A pull request that changes what a `generate_*` job writes names the jobs that must be rerun, in order, before the change can deploy.
- A change to an artifact's schema updates `scripts/fixture.py` and the artifact's prose in `docs/dev/reference.rst` together.
- A new artifact that is not one row per galaxy is added to `COPIED` in `scripts/dev_subset.py`, or given its own cut.
- Code that reads an artifact larger than one batch reads it in batches into preallocated arrays, and never holds a second copy of it.
