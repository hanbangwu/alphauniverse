from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.login.password import PASSWORD_COOKIE
from app.responses import RevalidatedRoute

PASSWORD_COOKIE_AGE = 30 * 24 * 60 * 60

router = APIRouter(tags=["login"], route_class=RevalidatedRoute)


@router.get("/login", status_code=204)
def check_login() -> None:
    pass


@router.post("/login")
def login(
    bearer: Annotated[HTTPAuthorizationCredentials, Depends(HTTPBearer())],
    response: Response,
) -> None:
    response.set_cookie(
        PASSWORD_COOKIE,
        bearer.credentials,
        max_age=PASSWORD_COOKIE_AGE,
        httponly=True,
        secure=True,
        samesite="lax",
    )
