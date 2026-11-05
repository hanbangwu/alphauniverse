---
paths:
  - "app/**"
  - "core/**"
  - "pipeline/**"
  - "scripts/**"
  - "tests/**"
  - ".github/workflows/**"
  - "web/src/**"
---

# Docs upkeep

- A change to a file below re-reads the pages in `docs/` it feeds and, in the same pull request, corrects any statement, formula, figure or caption it makes wrong:

  | Code                                                                                                                                         | Pages                                                                          |
  | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
  | `pipeline/encode.py`, `pipeline/predictions.py`, `pipeline/config.py`, `pipeline/schemas.py`, `core/config.py`, `core/stored_predictions.py` | `concepts.rst`, `glossary.rst`, `dev/reference.rst`                            |
  | `pipeline/grams.py`, `pipeline/embeddings.py`, `pipeline/aion_predictions.py`                                                                | `dev/reference.rst`, `glossary.rst`                                            |
  | `pipeline/observations.py`, `pipeline/observed_grams.py`                                                                                     | `dev/reference.rst`, `glossary.rst`                                            |
  | `pipeline/properties.py`                                                                                                                     | `dev/reference.rst`, `glossary.rst`                                            |
  | `pipeline/parametric_umap.py`, `pipeline/maps.py`                                                                                            | `methods/map.rst`, `dev/reference.rst`                                         |
  | `app/pql.py`, `core/token_search.py`                                                                                                         | `methods/similarity-search.rst`, `methods/query-maps.rst`, `dev/reference.rst` |
  | `app/text_search.py`, `core/text_model.py`, `pipeline/text_pairs.py`, `pipeline/alignment.py`                                                | `methods/text-search.rst`, `dev/reference.rst`                                 |
  | `pipeline/text_views.py`, `pipeline/text_regression.py`                                                                                      | `dev/reference.rst`, `glossary.rst`                                            |
  | `app/main.py`, `app/galaxy.py`                                                                                                               | `dev/reference.rst`, `dev/behaviour.rst`                                       |
  | `core/dataset.py`, `scripts/alphauniverse_cosmos.py`                                                                                         | `index.rst`                                                                    |
  | `app/modal_app.py`, `pipeline/modal_app.py`, `core/modal_resources.py`                                                                       | `dev/reference.rst`, `dev/how-to.rst`                                          |
  | `scripts/benchmarks/**`                                                                                                                      | `dev/benchmarks.rst`, `benchmark_plots.py`, `dev/how-to.rst`                   |
  | `tests/**`, `scripts/fixture.py`, `.github/workflows/**`                                                                                     | `dev/reference.rst`, `dev/how-to.rst`                                          |
  | `web/src/**`                                                                                                                                 | `using.rst`, `dev/behaviour.rst`, the matching `methods/` page                 |

- A term a page introduces gets a `glossary.rst` entry, and the page links it with `:term:`.
- A schematic whose subject changes is redrawn in the same pull request.
