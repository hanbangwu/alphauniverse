"""Write the API's OpenAPI schema to frontend/openapi.json."""

import json
from pathlib import Path

from app.main import app

TARGET = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


if __name__ == "__main__":
    TARGET.write_text(json.dumps(app.openapi(), indent=2) + "\n")
