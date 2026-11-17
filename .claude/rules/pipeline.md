---
paths:
  - "app/dataset/dataset.py"
  - "app/encoding/store.py"
  - "app/encoding/build/encode.py"
  - "app/predictions/store.py"
  - "app/predictions/build/predictions.py"
  - "app/predictions/build/aion_predictions.py"
  - "app/predictions/build/observations.py"
  - "app/predictions/build/grams.py"
  - "app/predictions/build/embeddings.py"
  - "app/predictions/build/observed_grams.py"
  - "app/text_search/text_search.py"
  - "app/text_search/build/alignment.py"
  - "app/text_search/build/text_views.py"
  - "app/text_search/build/text_regression.py"
  - "app/projection/build/parametric_umap.py"
  - "app/projection/build/properties.py"
  - "modal_app.py"
  - "pyproject.toml"
  - "scripts/alphauniverse_cosmos.py"
---

# Pipeline

- A pull request that changes what a `generate_*` job writes names the jobs that must be rerun, in order, before the change can deploy.
- A change to an artifact's schema updates `scripts/fixture.py` and the artifact's prose in `docs/dev/reference.rst` together.
- A new artifact that is not one row per galaxy is added to `COPIED` in `scripts/dev_subset.py`, or given its own cut.
- Code that reads an artifact larger than one batch reads it in batches into preallocated arrays, and never holds a second copy of it.
