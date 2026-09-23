---
paths:
  - "app/main.py"
  - "app/search.py"
  - "app/cutouts.py"
  - "app/spectra.py"
  - "app/config.py"
  - "modal_app.py"
---

# Backend

- The server only reads and never generates/modifies artifacts.
- Anything a request would otherwise load first is loaded in `lifespan`.
- Every endpoint is a `GET` with a typed response: a pydantic model for JSON, or a `responses=` entry naming the content type for binary bodies, so the generated client can type it.
- A deliberate error is an `HTTPException` with a message, and a route that can return 404 declares `NOT_FOUND`.
- A change to a route or response model regenerates `frontend/openapi.json` and updates the frontend in the same pull request, and updates the endpoint table in `docs/architecture.md`.
- Constants used by more than one module live in `app/config.py`.
