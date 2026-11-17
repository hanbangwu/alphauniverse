import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from app.config import (
    galaxy_count,
)

ARROW = "application/vnd.apache.arrow.stream"


@pytest.mark.usefixtures("identified")
def test_positions_are_in_galaxy_order(client: TestClient) -> None:
    response = client.get("/positions")
    positions = pa.ipc.open_stream(response.content).read_all()

    assert response.headers["content-type"] == ARROW
    assert positions.num_rows == galaxy_count()
    galaxy = client.get("/galaxy/6").json()
    assert positions.column("ra")[6].as_py() == galaxy["ra"]
    assert positions.column("dec")[6].as_py() == galaxy["dec"]
