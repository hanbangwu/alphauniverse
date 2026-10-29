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

  | Code                                                     | Pages                                                                          |
  | -------------------------------------------------------- | ------------------------------------------------------------------------------ |
  | `app/encode.py`, `app/predictions.py`, `app/config.py`   | `concepts.rst`, `glossary.rst`, `dev/reference.rst`                            |
  | `app/grams.py`, `app/embeddings.py`, `app/templates.py`  | `dev/reference.rst`, `glossary.rst`                                            |
  | `app/observations.py`, `app/observed_grams.py`           | `dev/reference.rst`, `glossary.rst`                                            |
  | `app/properties.py`                                      | `dev/reference.rst`, `glossary.rst`                                            |
  | `app/parametric_umap.py`, `app/maps.py`                  | `methods/map.rst`, `dev/reference.rst`                                         |
  | `app/pql.py`                                             | `methods/similarity-search.rst`, `methods/query-maps.rst`, `dev/reference.rst` |
  | `app/text_search.py`, `app/alignment.py`                 | `methods/text-search.rst`, `dev/reference.rst`                                 |
  | `app/main.py`                                            | `dev/reference.rst`, `dev/behaviour.rst`                                       |
  | `app/dataset.py`, `scripts/alphauniverse_cosmos.py`      | `index.rst`                                                                    |
  | `modal_app.py`                                           | `dev/reference.rst`, `dev/how-to.rst`                                          |
  | `scripts/benchmarks/**`                                  | `dev/benchmarks.rst`, `benchmark_plots.py`, `dev/how-to.rst`                   |
  | `tests/**`, `scripts/fixture.py`, `.github/workflows/**` | `dev/reference.rst`, `dev/how-to.rst`                                          |
  | `frontend/src/**`                                        | `using.rst`, `dev/behaviour.rst`, the matching `methods/` page                 |

- A term a page introduces gets a `glossary.rst` entry, and the page links it with `:term:`.
- A schematic whose subject changes is redrawn in the same pull request.
