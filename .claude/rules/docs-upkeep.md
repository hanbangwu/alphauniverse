---
paths:
  - "app/**"
  - "modal_app.py"
  - "benchmarks/**"
  - "scripts/**"
  - "tests/**"
  - ".github/workflows/**"
  - "web/src/**"
---

# Docs upkeep

- A change to a file below re-reads the pages in `docs/` it feeds and, in the same pull request, corrects any statement, formula, figure or caption it makes wrong:

  | Code                                                                                    | Pages                                                                          |
  | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
  | `app/encoding/build/encode.py`, `app/predictions/build/predictions.py`, `app/config.py` | `concepts.rst`, `glossary.rst`, `dev/reference.rst`                            |
  | `app/predictions/build/{grams,embeddings,aion_predictions}.py`                          | `dev/reference.rst`, `glossary.rst`                                            |
  | `app/predictions/build/{observations,observed_grams}.py`                                | `dev/reference.rst`, `glossary.rst`                                            |
  | `app/projection/build/properties.py`                                                    | `dev/reference.rst`, `glossary.rst`                                            |
  | `app/projection/build/{parametric_umap,maps}.py`                                        | `methods/map.rst`, `dev/reference.rst`                                         |
  | `app/search/pql.py`, `app/predictions/whole_mode.py`                                    | `methods/similarity-search.rst`, `methods/query-maps.rst`, `dev/reference.rst` |
  | `app/text_search/text_search.py`, `app/text_search/build/alignment.py`                  | `methods/text-search.rst`, `dev/reference.rst`                                 |
  | `app/text_search/build/{text_views,text_regression}.py`                                 | `dev/reference.rst`, `glossary.rst`                                            |
  | `app/main.py`, `app/responses.py`, `app/*/routes.py`, `app/login/password.py`           | `dev/reference.rst`, `dev/behaviour.rst`                                       |
  | `app/dataset/dataset.py`, `scripts/alphauniverse_cosmos.py`                             | `index.rst`                                                                    |
  | `modal_app.py`                                                                          | `dev/reference.rst`, `dev/how-to.rst`                                          |
  | `benchmarks/**`                                                                         | `dev/benchmarks.rst`, `benchmark_plots.py`, `dev/how-to.rst`                   |
  | `tests/**`, `scripts/fixture.py`, `scripts/dev_subset.py`, `.github/workflows/**`       | `dev/reference.rst`, `dev/how-to.rst`                                          |
  | `web/src/**`                                                                            | `using.rst`, `dev/behaviour.rst`, the matching `methods/` page                 |

- A term a page introduces gets a `glossary.rst` entry, and the page links it with `:term:`.
- A schematic whose subject changes is redrawn in the same pull request.
