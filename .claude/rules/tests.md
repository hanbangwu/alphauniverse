---
paths:
  - "tests/**/*"
  - "scripts/fixture.py"
---

# Tests

- `tests/` tests the `app/` code against the fixture tree, and nothing else: benchmarks have no tests. It is hermetic: no Modal, no network, no production artifacts. Checks on the real artifacts do not belong here.
- One test file per `app/` source file, at the same path under `tests/` (`app/search/pql.py` is tested by `tests/search/test_pql.py`), testing its functions and loaders directly.
- A module that runs AION or its codecs is tested with them built with random weights from configs vendored in `tests/`; no test downloads weights. The suite runs with the `build` group, as CI runs it.
- Each module's `test_routes.py` holds only the HTTP contract of its routes: status codes, content types and response shapes.
- A test earns its place by catching a plausible bug in this repository's code that no other test catches. If no such bug comes to mind, do not write it.
- Test what callers rely on: contracts, invariants (ordering, id layout, determinism), agreement with a reference implementation, and regressions. Assert the behavior that matters, not every observable property.
- Do not test a library's behavior, _e.g._ FastAPI's validation, Starlette's file responses or `save_pretrained`.
- Take expected values from the artifacts the test reads, not from how `scripts/fixture.py` generated them.
- Name a test as a sentence stating the behavior: `test_rows_out_of_galaxy_order_are_rejected`.
