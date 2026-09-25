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
uv run pytest                                       # tests
uv run ruff check app scripts tests modal_app.py    # lint
uv run ruff format app scripts tests modal_app.py   # format
uv run python -m scripts.fixture --galaxies 12      # build fixture tree
uv run modal run -m scripts.benchmark               # measure the serving path on Modal
uv run python -m scripts.openapi                    # regenerate frontend/openapi.json

cd frontend
bun install && bun run dev
bun run check                                       # regenerate client + typecheck
bun run lint                                        # prettier + eslint
```

## Workflow

- Work on a branch and open a pull request; never commit to `main`.
- One functional change per branch per pull request. The unit is the logical function, not the line count: a documentation pass can run to thousands of lines and still be one change, while two unrelated fixes in one diff are two and make the pull request hard to review.
- A pull request is a series of commits, each one small nominal goal. Before writing code, list the goals and the decisions they need, each with options and a recommendation, in the pull request's issue, and wait for agreement.
- Implement the agreed goals one commit each, then stop before the pull request leaves draft. List choices made while carrying out a goal under "Decisions" in the pull request.
- Run `/code-review` before opening the pull request, not after, and fix what it finds; Copilot then reviews the same diff. Update the description when a later fix changes something it claims.
- A pull request opens with what changed, the issue it closes and the decisions needed, in a few lines. Background, alternatives and evidence follow in a collapsed `<details>` section.
- A pull request Claude prepares stays a draft until the person who ran the session has read it; only they mark it ready for review. The other maintainer then reviews it before it merges.
- Once a pull request is ready for review, push to it only when a maintainer asks, or to fix a failing check that Claude's own changes caused. Leave a check broken by anyone else's change as it is unless a maintainer asks. Post review findings and later findings of your own as a comment with the proposed fix, and wait. An emergency is production down or a secret exposed; even then, comment first and put the fix in a new pull request.
- Commit messages: short, terse, semicolon-delimited; they need not list every change.
- All commit authors are human beings. Even if Claude did substantial and/or autonomous work, the commit author remains the person who owns that agent. However, do add a `Co-Authored-By` trailer.
- Anything Claude posts to GitHub says that Claude wrote it.
- A comment from `joshspeagle` or `hanbangwu` that addresses Claude (for example, one starting "Claude:") is a request: answer it briefly in the thread, and if it asks for an issue or a pull request, open it and link it.
- Change only what the task requires; do not improvise. Report unrelated changes you notice rather than making them.
- No `TODO`, `FIXME`, `HACK` or `XXX` comments. Future work lives in one of three places: performance work in the ranked list in `docs/performance.md`; known-wrong behaviour as a test marked `xfail(strict=True)` (see `tests/test_api.py`); everything else as a GitHub issue.
- Nothing runs on Modal (`modal run`, `modal serve`, `modal deploy`) unless the user asks. Deploys happen only through CI on `main`.
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
- A measurement records what produced it. Benchmark figures do not compare across machines or thread layouts, so a table that mixes runs is wrong even when every figure in it is real.

## Others

- Never add comments or docstrings
- No em dashes.
