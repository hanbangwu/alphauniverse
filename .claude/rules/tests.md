---
paths:
  - "tests/**/*"
  - "scripts/fixture.py"
---

# Tests

- `tests/` tests the code against the fixture tree. It is hermetic: no Modal, no network, no production artifacts. Checks on the real artifacts do not belong here.
- One file per `app/` module, testing its functions and loaders directly.
- A module that imports torch is tested with AION and its codecs built with random weights from configs vendored in `tests/`; no test downloads weights. Its test file calls `pytest.importorskip("torch")` before importing anything that needs torch, so the default suite runs without the `build` group, and a CI job that installs the group runs it.
- `test_api.py` holds only the HTTP contract: status codes, content types and response shapes.
- A test earns its place by catching a plausible bug in this repository's code that no other test catches. If no such bug comes to mind, do not write it.
- Test what callers rely on: contracts, invariants (ordering, id layout, determinism), agreement with a reference implementation, and regressions. Assert the behaviour that matters, not every observable property.
- Do not test a library's behaviour, _e.g._ FastAPI's validation, Starlette's file responses or `save_pretrained`.
- Take expected values from the artifacts the test reads, not from how `scripts/fixture.py` generated them.
- Name a test as a sentence stating the behaviour: `test_rows_out_of_galaxy_order_are_rejected`.
