---
paths:
  - "app/main.py"
  - "app/responses.py"
  - "app/*/routes.py"
  - "app/login/password.py"
  - "app/dataset/dataset.py"
  - "app/encoding/store.py"
  - "app/predictions/store.py"
  - "app/predictions/whole_mode.py"
  - "app/search/pql.py"
  - "app/text_search/text_search.py"
  - "app/config.py"
  - "modal_app.py"
---

# Backend

- The server only reads and never generates/modifies artifacts.
- Anything a request would otherwise load first is loaded in `lifespan`, except the Hugging Face dataset, text search's WeMM-Embedding and vectors, and a galaxy's agreement matrix, which load on first use.
- Every endpoint has a typed response: a pydantic model for JSON, or a `responses=` entry naming the content type for binary bodies, so the generated client can type it.
- A deliberate error is an `HTTPException` with a message.
- A change to a route or response model regenerates `web/openapi.json`, makes the smallest web app change that keeps `bun run check` passing, all in the same pull request.
- Constants used by more than one module live in `app/config.py`.
