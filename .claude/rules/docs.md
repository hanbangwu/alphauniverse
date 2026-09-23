---
paths:
  - "docs/**"
  - "README.md"
  - "CLAUDE.md"
  - ".claude/rules/**"
---

# Docs

- Docs are for someone new, concise, and state only what the code does now.
- Each fact has one home: before adding to a doc, re-read it and cut what the addition duplicates. The homes are:
  - `README.md`: what the platform does, the dataset, and how to run it.
  - `docs/architecture.md`: how the pieces connect, the artifact roles and the endpoints.
  - `docs/pipeline.md`: the build jobs and the artifact schemas.
  - `docs/performance.md`: the cost model, measured figures, ranked candidate work and scaling ceilings.
  - `docs/testing.md`: the test, benchmark and CI commands.
- A doc changes in the same pull request as the code it describes.
- A rule goes in the narrowest place it applies: a path-scoped file in `.claude/rules/` when it concerns certain files, `CLAUDE.md` only when it applies to the whole repository.
