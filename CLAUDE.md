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
bun install && bun run dev                          # needs PUBLIC_API_URL, see .env.example
bun run check                                       # regenerate client + typecheck
bun run lint                                        # prettier + eslint
```

## Workflow

- Always work on a branch and never commit or push to `main`.
- All changes land as a pull request.

## Future Work Documentation

- Do not add `TODO`, `FIXME`, `HACK` or `XXX` comments anywhere in the codebase. An inline marker is tracked by nothing and goes stale without anyone noticing.
- Known future work lives in three places:

1. Performance work - the ranked list in `docs/performance.md`, ordered by measured cost and re-ordered when a measurement changes it.
2. Behaviours known to be wrong - a test marked `xfail(strict=True)`, so the fix cannot land without someone removing the marker. See `tests/test_api.py`.
3. Everything else - a GitHub issue.

## Coding Guidelines

- Prioritize code correctness and clarity over speed and efficiency.
- Only write comments to explain "why" the code is written in some non-obvious or tricky way. Do not write organizational comments or summaries.
- Each line in docstrings should have a maximum of 80 characters. If exceeded, it means you are too verbose.
- Prefer implementing functionality in existing files unless it is a new logical component.
- Always use library API calls rather than hand-rolling. If a suitable library is not installed, ask the user for permission to install it.
- Only write helpers when a function becomes too long and/or the helper would be used in multiple locations.
- Never write no-ops. This includes: passing the default value of a parameter into a parameter, useless declarations (_e.g._ if an array is already in `float32`, don't declare it as `float32` again (unless it's convention)), or useless guards/conversions (_e.g._ if an array is already in `float32`, and you know that it always will be, don't explicitly convert it to `float32`). The exception is typing constants.
- Never write linter ignore rules as comments, configs, or any other expressions.
- Use full words for variable names (no abbreviations like "q" for "queue").
- Avoid writing anti-patterns. Libraries should usually be used the way they are intended to. _e.g._ if the documentation specifies a way to complete a task, then follow documentation examples where possible and deviate from documentation and documentation examples only for necessary functional reasons.
- Always read the documentation when unsure. Report when your empirical tests (which I don't prefer over documentation nor the other way around) contradict documentation.
- Always think about why a value is passed into a parameter of a library call. Usually, the default is okay unless you have an explicit, significant reason to use another value.
- Always change only what the task requires. If you notice an unrelated, potentially desirable change, report it.
- Always write the minimum amount of code needed without sacrificing behaviour, performance, etc.
- A function should serve one purpose and should be reasonably short.
- Keep commit message short and terse. Delimit using semicolons. It's okay to not mention all changes.
- Prefer removing to adding. A change that deletes a concept is usually better than one that adds a flag.
- Documentation states what the code does now. Not what it used to do, not what it might do, not which alternatives were rejected cuz no one cares.
- Each fact has one home. If something is documented in two places, one of them is already wrong.
- When adding to a doc, re-read the whole file first and cut what the addition duplicates. Otherwise they accumulate duplicates.
- Working notes stay out of the repository. Scratch analysis, session logs and investigation write-ups are thrown away, not committed.
- Tests assert the behaviour that matters, not every property that happens to be observable.
- Mannered prose substitutes metaphor and flourish for direct statement: "a dial worth turning" instead of "a parameter worth varying", "this point earns its keep" instead of "this point still matters". The phrases display the writer rather than convey the idea, and readers can tell. They are also imprecise: a metaphor carries connotations the writer did not choose and cannot control. When a literal phrase is available, use it.
- I personally think the use of em dashes should be a misdemeanour. I think you know what to do with that information.
- Use lists and headings when asked for them, or when the content is multifaceted enough that they aid clarity. Not by default.
- If minimal formatting is requested, use none: no bullets, headings, lists or bold.
- Finally, always ask when unsure about anything at all.

Specific for this project: don't edit `frontend/src/components/ui`.
