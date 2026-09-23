---
paths:
  - "tests/**/*"
  - "scripts/fixture.py"
---

# Writing tests

- `tests/` tests the code against the fixture tree. It is hermetic: no Modal, no network, no production artifacts. Checks on the real artifacts do not belong here.
- One file per `app/` module, testing its functions and loaders directly. `test_api.py` holds only the HTTP contract: status codes, content types and response shapes.
- Test what callers rely on: contracts, invariants (ordering, id layout, determinism), agreement with a reference implementation, and regressions. Assert the behaviour that matters, not every observable property.
- Test an error path only when the code checks it deliberately or the API promises it. Build the bad input inline in the test, not by editing the shared tree.
- Take expected values from the artifacts the test reads, not from how `scripts/fixture.py` generated them.
- No assertion that holds only because the fixture is small. One that needs production scale belongs in the benchmark or the checks on the real artifacts.
- Name a test as a sentence stating the behaviour: `test_rows_out_of_galaxy_order_are_rejected`.
