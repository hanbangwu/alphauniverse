# CLAUDE.md

## Layout

| Path           | What it is                                               |
| -------------- | -------------------------------------------------------- |
| `app/`         | Build pipeline and the FastAPI app                       |
| `modal_app.py` | Modal deployment                                         |
| `scripts/`     | Dataset build, OpenAPI export, test fixtures, benchmarks |
| `tests/`       | pytest suite                                             |
| `frontend/`    | Svelte app                                               |
| `docs/`        | Architecture, pipeline, performance, and testing notes   |

## Commands

```sh
uv run pytest                                       # test suite
uv run python -m scripts.fixture --galaxies 12      # build a fixture tree
uv run python -m scripts.benchmark --galaxies 32    # measure the search path
uv run python -m scripts.openapi                    # regenerate frontend/openapi.json

cd frontend
bun install && bun run dev
bun run check                                       # regenerate client + typecheck
bun run lint                                        # prettier + eslint
```

## Workflow

- Work on a branch and open a pull request; never commit to `main`.
- One functional change per branch per pull request. The unit is the logical function, not the line count: a documentation pass can run to thousands of lines and still be one change, while two unrelated fixes in one diff are two and make the pull request hard to review.
- Run `/code-review` before opening the pull request, not after, and fix what it finds; Copilot then reviews the same diff. Update the description when a later fix changes something it claims.
- Commit messages: short, terse, semicolon-delimited; they need not list every change.
- Change only what the task requires; do not improvise. Report unrelated changes you notice rather than making them.
- Never edit `frontend/src/lib/components/ui`.
- No `TODO`, `FIXME`, `HACK` or `XXX` comments. Future work lives in one of three places: performance work in the ranked list in `docs/performance.md`; known-wrong behaviour as a test marked `xfail(strict=True)` (see `tests/test_api.py`); everything else as a GitHub issue.
- Working notes (scratch analysis, session logs, write-ups) are not committed.
- Ask when unsure about anything: a new file or not, leanness versus performance, installing a library.

## Code

- Write the minimum code that is correct and clear.
- Prefer removing to adding; deleting a concept is better than adding a flag.
- Use library APIs the way their documentation intends, and read the docs when unsure. Don't hand-roll what a library provides.
- Leave parameters at their defaults unless there is an explicit, significant reason not to.
- No no-ops: passing a parameter its default, redeclaring a type a value already has, or guarding a case that cannot happen. Typing constants is fine.
- One purpose per function, one group of things per file. Add a helper only when a function is too long or the helper is reused.
- No linter-ignore rules, as comments, config or otherwise.
- Tests assert the behaviour that matters, not every observable property.
- A measurement records what produced it. Benchmark figures do not compare across machines or thread layouts, so a table that mixes runs is wrong even when every figure in it is real.

## Comments and docs

- No comments or docstrings unless the code is unconventional enough to need clarification.
- Docs are for someone new, concise, and state only what the code does now. Each fact has one home: before adding to a doc, re-read it and cut what the addition duplicates.
- Prefer literal phrasing to metaphor and flourish: "a parameter worth varying", not "a dial worth turning".
- No em dashes.
