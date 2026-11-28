from typing import Annotated

from fastapi import Depends, HTTPException

from alphauniverse.encoding.store import rows


def located(galaxy: str) -> int:
    try:
        return int(rows([galaxy])[0])
    except ValueError:
        raise HTTPException(404, f"no galaxy {galaxy}") from None


Row = Annotated[int, Depends(located)]
