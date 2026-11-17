from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response

from app.config import Projection, artifact
from app.responses import BINARY_OCTET, RevalidatedRoute, file

router = APIRouter(tags=["projection"], route_class=RevalidatedRoute)


@router.get(
    "/projections/{projection}",
    response_class=FileResponse,
    responses={200: {"content": BINARY_OCTET}},
)
def get_projections(projection: Projection, request: Request) -> Response:
    return file(artifact(f"{projection}_points"), request)


@router.head("/projections/{projection}", include_in_schema=False)
def head_projections(projection: Projection, request: Request) -> Response:
    return get_projections(projection, request)


@router.get(
    "/galaxies/property",
    response_class=FileResponse,
    responses={200: {"content": BINARY_OCTET}},
)
def get_properties(request: Request) -> Response:
    return file(artifact("properties"), request)


@router.head("/galaxies/property", include_in_schema=False)
def head_properties(request: Request) -> Response:
    return get_properties(request)
