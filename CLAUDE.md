# CLAUDE.md

## Layout

| Path           | What it is                                               |
| -------------- | -------------------------------------------------------- |
| `app/`         | Build pipeline and the FastAPI app                       |
| `modal_app.py` | Modal deployment                                         |
| `scripts/`     | Dataset build, OpenAPI export, test fixtures, benchmarks |
| `tests/`       | pytest suite                                             |
| `frontend/`    | Svelte app                                               |
| `docs/`        | Sphinx documentation and the benchmark reports           |

## Commands

### Backend

```sh
uv run --group build pytest                                 # tests
uv run ruff check app scripts tests modal_app.py            # lint
uv run ruff format app scripts tests modal_app.py           # format
uv run python -m scripts.fixture                            # build fixture tree
uv run modal run -m scripts.benchmarks.search_performance   # time similarity search on Modal
uv run modal run -m scripts.benchmarks.search_quality       # measure similarity search on Modal
uv run modal run -m scripts.benchmarks.backend_performance  # time the HTTP endpoints on Modal
uv run modal run -m scripts.benchmarks.projection_quality   # measure projection quality on Modal
uv run modal run -m scripts.benchmarks.text_search_quality  # measure text search on Modal
uv run --group build modal run -m scripts.benchmarks.compression_quality  # measure prediction storage on Modal
uv run python -m scripts.openapi                            # regenerate frontend/openapi.json
```

### Docs

```sh
uv run --group docs make -C docs html  # build the Sphinx site into docs/_build/html
```

### Frontend

```sh
cd frontend
bun install && bun run dev
bun run check               # regenerate client + typecheck
bun run lint                # prettier + eslint
```

## Working

- Change only what the task requires. Report unrelated problems you notice instead of fixing them.
- Ask before editing, not midway, when unsure: a new file, a new library, leanness versus performance. Ask when rules conflict.
- Regenerate generated files (`openapi.json`, `package.json`) with their commands, not by hand.

## Modal

- Nothing runs on Modal (`modal run`, `modal serve`, `modal deploy`) and no load or cold-start traffic goes to the deployed app unless a maintainer asks.
- Deploys happen only through CI: `main` to the default environment, and any branch to the `dev` environment when the CI/CD workflow is run by hand. `--env dev` runs a job against the `dev` app and volume.

## Git

- Branch every pull request from `main`; never commit to `main`. If one change needs another, wait for it to merge.
- One logical change per branch, whatever its line count: two unrelated fixes are two branches.
- Pull before each commit and build on a maintainer's edits. Never force-push, rebase or reset a branch a human has committed to, or push to one someone is merging.
- Commit messages are short and semicolon-delimited. They also need not list every change.
- The commit author is the human who runs the session with a `Co-Authored-By` trailer for Claude.

## Pull requests

- A pull request is a series of commits, one small goal each. Implement them, then stop while it is a draft. List choices made along the way under "Decisions".
- Before opening, run `/simplify`, then `/code-review`, and fix what they find. Request a Copilot review only when it changes search results, a route or response model, an artifact's schema or a `generate_*` job.
- It updates anything it makes stale in `README.md`, `docs/`, `CLAUDE.md` and `.claude/rules/`, and fixes errors there on the subject it changes. A code pull request found to skip this gets the fixes proposed in a docs pull request.
- The description opens with what changed, the issue it closes and the decisions needed, in a few lines; background, alternatives and evidence go in a collapsed `<details>`. Update it when a later fix changes what it claims.
- Use only GitHub's default labels: `bug` for wrong behaviour; `enhancement` for features, performance, tests, CI and refactors; `documentation` when docs or instructions are the main change; `question` while waiting on an answer; `duplicate` or `wontfix` when it closes for either reason.

## Review and merge

- A draft stays a draft until the person who ran the session has read it. They mark it ready, or Claude does and merges it once they comment that they are happy to merge.
- Minor bug fixes, test-only changes, docs changes and backend changes `joshspeagle` approves in a comment merge without the other maintainer's review. Anything else, and anything touching `frontend/`, waits for it.
- Claude monitors every pull request it opens until it merges or closes (CI, comments, merge conflicts), checking in about an hour after opening.
- Once a pull request is ready for review, push to it only when a maintainer asks, to fix a check Claude's own change broke, or to address Copilot's comments while no maintainer has pushed to the branch. Post anything else as a comment with the proposed fix, and wait.
- In an emergency (production down, a secret exposed), comment first and put the fix in a new pull request.

## GitHub comments

- Declare anything that is written by Claude only as being from Claude.
- A comment addressing Claude (for example, starting "Claude:") is a request: answer briefly in the thread, and open and link any issue or pull request it asks for.

## Code

- Write the minimum code that is correct and clear. Prefer removing to adding; deleting a concept beats adding a flag.
- Before writing code, stop at the first that serves: the change is not needed; the codebase, the standard library, the platform (browser, Svelte, FastAPI, pyarrow) or an installed dependency already does it.
- Use library APIs as their documentation intends; read it when unsure. Report when a test contradicts it; neither outranks the other.
- Leanness never removes request validation, data-loss handling, security checks or accessibility.
- Keep names (for methods, variables, etc) human-readable.
- Leave parameters at their defaults unless there is a significant reason not to.
- No no-ops: passing a parameter its default, redeclaring a type a value already has, or guarding a case that cannot happen. Typing constants is fine.
- One purpose per function, one group of things per file. Add a helper only when a function is too long or the helper is reused.
- No linter-ignore rules, except excluding generated or vendored paths and settings a tool's documentation recommends.
- No comments or docstrings, except in generated, vendored or scaffold files. Why a change was made goes in its issue or pull request; how the code works goes in `docs/`.

## Measurements

- State only measured figures, or figures computed from them with the computation shown and consistent units.
- The fixture tests correctness; never use it to measure or extrapolate production behaviour.
- A measurement records what produced it. Figures from different machines or thread layouts do not compare, so a table that mixes runs is wrong.

## Writing

- Spartan: what is true now, in the fewest words that stay exact. No restatement, hedges, needless background or lists of what did not change.
- Literal over metaphor: "a parameter worth varying", not "a dial worth turning".
- No em dashes.
- Session replies lead with the answer, then the reason, then the next step. Warnings and irreversible actions get full sentences.
