import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyCookie, HTTPAuthorizationCredentials, HTTPBearer

PASSWORD_COOKIE = "password"

cookie_password = APIKeyCookie(name=PASSWORD_COOKIE, auto_error=False)
bearer_password = HTTPBearer(auto_error=False)


def require_password(
    cookie: Annotated[str | None, Depends(cookie_password)],
    bearer: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_password)],
) -> None:
    supplied = bearer.credentials if bearer else cookie
    expected = os.environ["ACCESS_PASSWORD"]
    if supplied is None or not secrets.compare_digest(
        supplied.encode(), expected.encode()
    ):
        raise HTTPException(401, "Password required", {"WWW-Authenticate": "Bearer"})
