---
paths:
  - "app/**"
  - "modal_app.py"
  - "scripts/**"
  - "tests/**"
  - ".github/workflows/**"
  - "frontend/src/**"
---

# Docs upkeep

- A change to a file below re-reads the pages in `docs/` it feeds and, in the same pull request, corrects any statement, formula, figure or caption it makes wrong:

  | Code                                                                     | Pages                                                         |
  | ------------------------------------------------------------------------ | ------------------------------------------------------------- |
  | `app/encode.py`, `app/predictions.py`, `app/config.py`                   | `concepts.rst`, `glossary.rst`, `reference.rst`               |
  | `app/grams.py`, `app/embeddings.py`                                      | `reference.rst`, `glossary.rst`                               |
  | `app/parametric_umap.py`                                                 | `methods/map.rst`                                             |
  | `app/pql.py`                                                             | `methods/similarity-search.rst`, `methods/query-maps.rst`     |
  | `app/text_search.py`, `app/alignment.py`                                 | `methods/text-search.rst`                                     |
  | `app/main.py`                                                            | `reference.rst`, `using.rst`                                  |
  | `app/dataset.py`, `scripts/alphauniverse_cosmos.py`                      | `index.rst`                                                   |
  | `modal_app.py`                                                           | `reference.rst`, `how-to.rst`                                 |
  | `scripts/benchmarks/**`                                                  | `benchmarks.rst`, `benchmark_plots.py`, `how-to.rst`          |
  | `tests/**`, `scripts/fixture.py`, `.github/workflows/**`                 | `reference.rst`, `how-to.rst`                                 |
  | `frontend/src/**`                                                        | `using.rst`, the matching page in `methods/`                  |

- A term a page introduces gets a `glossary.rst` entry, and the page links it with `:term:`.
- A schematic whose subject changes is redrawn in the same pull request.
