# CLAUDE.md

## Layout

| Path       | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `app/`     | Build pipeline and the FastAPI app                           |
| `scripts/` | Dataset build, OpenAPI, dev subset, test fixture, benchmarks |
| `tests/`   | pytest suite                                                 |
| `web/`     | Web app                                                      |
| `docs/`    | Sphinx documentation and benchmark reports                   |

## Commands

### Backend

```sh
uv run --group build pytest                        # tests
uv run ruff check app scripts tests modal_app.py   # lint
uv run ruff format app scripts tests modal_app.py  # format
uv run python -m scripts.dev_subset                # build the dev subset on Modal and download it
uv run python -m scripts.openapi                   # regenerate web/openapi.json
```

### Web

```sh
cd web
bun install && bun run dev
bun run check
bun run lint
```

### Benchmarks

```sh
uv run modal run -m scripts.benchmarks.search_performance                     # time similarity search on Modal
uv run modal run -m scripts.benchmarks.search_quality                         # measure similarity search on Modal
uv run modal run -m scripts.benchmarks.backend_performance                    # time the HTTP endpoints on Modal
uv run modal run -m scripts.benchmarks.projection_quality                     # measure projection quality on Modal
uv run modal run -m scripts.benchmarks.text_search_quality                    # measure text search on Modal
uv run modal run -m scripts.benchmarks.text_search_sheet                      # draw text search's contact sheet on Modal
uv run --group build modal run -m scripts.benchmarks.compression_quality      # measure prediction storage on Modal
uv run modal run -m scripts.benchmarks.token_embedding_quality                # measure token embeddings on Modal
uv run --group build modal run -m scripts.benchmarks.aion_prediction_quality  # measure AION predictions against data on Modal
uv run modal run -m scripts.benchmarks.aion_prediction_embedding_quality      # measure AION prediction embeddings against data on Modal
uv run --group build modal run -m scripts.benchmarks.retrieval_quality        # measure retrieval from partial data on Modal
uv run --group build modal run -m scripts.benchmarks.probe_quality            # measure PROVABGS probes on the stored vectors on Modal
uv run --group build modal run -m scripts.benchmarks.redshift_quality         # measure AION's redshift without spectra on Modal
uv run --group build modal run -m scripts.benchmarks.neighbour_quality        # measure the stored vectors' neighbourhoods and maps on Modal
```

### Docs

```sh
uv run --group docs make -C docs html
```

## Rules

### Working

- Make the least amount of change while maintaining quality.
- Change only what the task requires. Report unrelated problems you notice instead of fixing them.
- When unsure, ask before editing: a new file, a new library, leanness versus performance.
- Ask when rules conflict.
- Regenerate generated files (`openapi.json`, `package.json`, etc) with their commands, not by hand.
- Always read documentation. Use library APIs as their documentation intends.
- Keep names (for methods, variables, etc) human-readable.
- Leave parameters at their defaults unless there is a significant reason not to.
- One purpose per function, one group of functions per file. Add a helper only when a function is too long or the helper is reused.
- No no-ops: passing a parameter its default, redeclaring a type a value already has, or guarding a case that cannot happen. Typing constants is fine.
- No hand-rolling: if appropriate, use library rather than writing our own code.
- No linter-ignore rules, except excluding generated or vendored paths and settings a tool's documentation recommends.
- Do not add comments or docstrings (leave ones from generated, vendored, or scaffold files alone).
- Maximize usage of OpenAPI. If it exists on both the backend and the web, it goes in OpenAPI.

### Modal

- Run things on Modal only with user permission.
- Deploys happen only through CI. `main` branch goes to the `main` environment. Other branches go to the `dev` environment.

### Git

- Branch every PR from `main`.
- Never autonomously commit to `main`.
- One logical change per branch. Note that modifications in a large number of lines might be just one logical change and vice versa.
- Never force-push, rebase, or reset a branch.
- Commit messages are short and semicolon-delimited. They also need not list every change.
- The commit author is the human who runs the session with a `Co-Authored-By` trailer for Claude.

### PRs

- A pull request is a series of commits, one small goal each. Implement them, then stop while it is a draft. List choices made along the way under "Decisions".
- Before opening, run `/simplify`, then `/code-review`, and fix what they find.
- Ensure docs are updated after changes in the same PR as the changes.
- The PR comment follows: which issue (if any), description of the changes (mixed with decisions made along the way). Update it accordingly if needed.

### Claude

- Autonomous Claude PRs stay draft until a human intervenes.
- Declare anything that is written by Claude only as being from Claude.
- Monitor comments from open PRs and issues.
- In an emergency (production down, a secret exposed), comment first and put the fix in a new PR.
- Do not create your own GitHub PR or issue tags.

## Documentation

- The documentation is both for the user to read and serves as internal documentation (readable by the public, but contains details on specific implementations and behind-the-scenes processes, decisions, and assumptions).
- State only measured figures, or figures computed from them with the computation shown and consistent units.
- Only use the full dataset (never the fixture or the dev subset) to measure production behaviour.
- Do not mix figures from different machines or thread layouts in the same table.

## Writing

- Spartan: what is true now, in the fewest words that stay exact. No restatement, hedges, needless background or lists of what did not change.
- Literal over metaphor: "a parameter worth varying", not "a dial worth turning".
- No em dashes.
