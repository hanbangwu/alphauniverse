---
paths:
  - "src/alphauniverse/**"
  - "modal_app.py"
  - "benchmarks/**"
  - "scripts/**"
  - "tests/**"
  - ".github/workflows/**"
  - "web/src/**"
---

# Docs upkeep

- A change to a file below re-reads the pages in `docs/` it feeds and, in the same pull request, corrects any statement, formula, figure or caption it makes wrong:

  | Code                                                                                                                              | Pages                                                                          |
  | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
  | `src/alphauniverse/encoding/build/encode.py`, `src/alphauniverse/predictions/build/predictions.py`, `src/alphauniverse/config.py` | `concepts.rst`, `glossary.rst`, `dev/reference.rst`                            |
  | `src/alphauniverse/predictions/build/{grams,embeddings,aion_predictions}.py`                                                      | `dev/reference.rst`, `glossary.rst`                                            |
  | `src/alphauniverse/predictions/build/observations.py`                                                                             | `dev/reference.rst`, `glossary.rst`                                            |
  | `server/projection/build/properties.py`                                                                                           | `dev/reference.rst`, `glossary.rst`                                            |
  | `server/projection/build/{parametric_umap,maps}.py`                                                                               | `methods/map.rst`, `dev/reference.rst`                                         |
  | `src/alphauniverse/search/pql.py`, `src/alphauniverse/predictions/whole_mode.py`                                                  | `methods/similarity-search.rst`, `methods/query-maps.rst`, `dev/reference.rst` |
  | `src/alphauniverse/text_search/text_search.py`, `src/alphauniverse/text_search/build/alignment.py`                                | `methods/text-search.rst`, `dev/reference.rst`                                 |
  | `src/alphauniverse/text_search/build/{text_views,text_regression}.py`                                                             | `dev/reference.rst`, `glossary.rst`                                            |
  | `server/main.py`, `server/responses.py`, `server/*/routes.py`, `server/login/password.py`                                         | `dev/reference.rst`, `dev/behavior.rst`                                        |
  | `src/alphauniverse/dataset/dataset.py`, `scripts/alphauniverse_cosmos.py`                                                         | `index.rst`                                                                    |
  | `modal_app.py`                                                                                                                    | `dev/reference.rst`, `dev/how-to.rst`                                          |
  | `benchmarks/**`                                                                                                                   | `dev/benchmarks.rst`, `benchmark_plots.py`, `dev/how-to.rst`                   |
  | `tests/**`, `scripts/fixture.py`, `scripts/dev_subset.py`, `.github/workflows/**`                                                 | `dev/reference.rst`, `dev/how-to.rst`                                          |
  | `web/src/**`                                                                                                                      | `using.rst`, `dev/behavior.rst`, the matching `methods/` page                  |

- A term a page introduces gets a `glossary.rst` entry, and the page links it with `:term:`.
- A schematic whose subject changes is redrawn in the same pull request.
