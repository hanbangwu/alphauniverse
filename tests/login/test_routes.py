from fastapi.testclient import TestClient

from app.login.password import PASSWORD_COOKIE


def test_the_login_check_answers_without_the_dataset(client: TestClient) -> None:
    assert client.get("/login").status_code == 204


def test_a_request_without_the_password_is_rejected(client: TestClient) -> None:
    assert TestClient(client.app).get("/login").status_code == 401


def test_a_wrong_bearer_password_is_rejected(client: TestClient) -> None:
    headers = {"Authorization": "Bearer wrong"}

    assert client.get("/login", headers=headers).status_code == 401


def test_a_wrong_cookie_is_rejected(client: TestClient) -> None:
    cookies = {PASSWORD_COOKIE: "wrong"}
    visitor = TestClient(client.app, cookies=cookies)

    assert visitor.get("/login").status_code == 401


def test_login_sets_a_cookie_that_authenticates(
    client: TestClient, password: str
) -> None:
    visitor = TestClient(client.app, base_url="https://testserver")
    headers = {"Authorization": f"Bearer {password}"}

    assert visitor.post("/login", headers=headers).status_code == 200
    assert visitor.get("/login").status_code == 204


def test_login_with_a_wrong_password_sets_no_cookie(client: TestClient) -> None:
    visitor = TestClient(client.app, base_url="https://testserver")
    headers = {"Authorization": "Bearer wrong"}

    assert visitor.post("/login", headers=headers).status_code == 401
    assert not visitor.cookies
