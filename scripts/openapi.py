"""Write the API's OpenAPI schema to frontend/openapi.json.

The frontend generates its typed client and zod schemas from this file, so it
must be regenerated whenever a route, model or docstring in app/main.py changes.
CI fails if the committed copy is stale.
"""

import json
from pathlib import Path

from app.main import app

TARGET = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


if __name__ == "__main__":
    TARGET.write_text(json.dumps(app.openapi(), indent=2) + "\n")
