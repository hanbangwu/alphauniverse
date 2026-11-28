import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize("url", ["/projections/mean", "/galaxy/fixture-0"])
@pytest.mark.usefixtures("identified")
def test_a_request_with_the_current_etag_is_not_modified(
    client: TestClient, url: str
) -> None:
    served = client.get(url)

    unchanged = client.get(url, headers={"If-None-Match": served.headers["etag"]})
    stale = client.get(url, headers={"If-None-Match": '"stale"'})

    assert (unchanged.status_code, unchanged.content) == (304, b"")
    assert (stale.status_code, stale.content) == (200, served.content)


@pytest.mark.usefixtures("identified")
def test_a_weak_etag_in_a_list_is_not_modified(client: TestClient) -> None:
    etag = client.get("/galaxy/fixture-0").headers["etag"]

    response = client.get(
        "/galaxy/fixture-0", headers={"If-None-Match": f'"stale", W/{etag}'}
    )

    assert response.status_code == 304
