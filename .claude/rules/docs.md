---
paths:
  - "docs/**"
  - "README.md"
  - "CLAUDE.md"
  - ".claude/rules/**"
---

# Docs

- `docs/` is a Sphinx project in reStructuredText. Its reader is an astronomer who knows little about AI: say what they see and what AION does, with math in place of prose where it is shorter; keep implementation detail to the reference.
- Docs are concise and state only what the code does now.
- Each fact has one home: before adding to a doc, re-read it and cut what the addition duplicates. The homes are:
  - `README.md`: a brief showcase of the features, how to start the site locally, and a link to the docs.
  - `docs/index.rst`: what alphaUniverse is and the dataset.
  - `docs/concepts.rst`: what AION does with a galaxy: tokens, embeddings, predictions.
  - `docs/glossary.rst`: one-line definitions of the terms the pages use, linked with `:term:`.
  - `docs/figures/`: schematics as SVG, drawn in `currentColor` and included with `.. raw:: html` so they follow the light and dark themes.
  - `docs/methods/`: how each feature works, one page each.
  - `docs/using.rst`: what a visitor can do and what each result means, without the layout.
  - `docs/benchmarks.rst`: what each benchmark measures, with its figures drawn from its report.
  - `docs/how-to.rst`: the commands.
  - `docs/reference.rst`: the pipeline, the artifacts, the HTTP API, and the tests and CI.
  - `docs/references.bib`: every paper the docs cite, cited with `:cite:p:` or `:cite:t:`, its title in double braces to keep its case; `docs/references.rst` lists them.
- Facts the code holds (schemas, routes) are rendered from it by Sphinx, not restated.
- Schematics explain a concept, never the UI's layout. A figure drawn from data is built when the docs build, from a committed report under `docs/benchmarks/` or the fixture, and names what produced it; it is never a screenshot or a committed image of a plot.
- A rule goes in the narrowest place it applies: a path-scoped file in `.claude/rules/` when it concerns certain files, `CLAUDE.md` only when it applies to the whole repository.
