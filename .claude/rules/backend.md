---
paths:
  - "app/**"
  - "core/**"
---

# Backend

- The server only reads and never generates/modifies artifacts.
- Anything a request would otherwise load first is loaded in `lifespan`, except the Hugging Face dataset, text search's WeMM-Embedding and vectors, and a galaxy's agreement matrix, which load on first use.
- Every endpoint is a `GET` with a typed response: a pydantic model for JSON, or a `responses=` entry naming the content type for binary bodies, so the generated client can type it.
- A deliberate error is an `HTTPException` with a message.
- A change to a route or response model regenerates `web/openapi.json`, makes the smallest web change that keeps `bun run check` passing, all in the same pull request; Hanbang reviews the web app part.
- Constants used by more than one `app/` module live in `app/config.py`; those `pipeline/` also uses live in `core/config.py`.
