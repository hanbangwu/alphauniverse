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
  - `docs/explanation.rst`: how each feature works.
  - `docs/benchmarks.rst`: what each benchmark measures, with its figures drawn from its report.
  - `docs/how-to.rst`: the commands.
  - `docs/reference.rst`: the pipeline, the artifacts, the HTTP API, and the tests and CI.
  - `docs/references.bib`: every paper the docs cite, cited with `:cite:p:` or `:cite:t:` and listed on `docs/references.rst`.
- Facts the code holds (schemas, routes) are rendered from it by Sphinx, not restated.
- A rule goes in the narrowest place it applies: a path-scoped file in `.claude/rules/` when it concerns certain files, `CLAUDE.md` only when it applies to the whole repository.
