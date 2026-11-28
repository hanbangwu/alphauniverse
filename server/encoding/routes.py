import numpy as np
from aion.modalities import LegacySurveyImage
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from alphauniverse.config import TOKEN_SURVEYS, ImageSurvey, SpectrumSurvey
from alphauniverse.encoding.store import tokens
from server.dataset.dataset import identity
from server.encoding.false_color import false_colors, image_lattice, spectrum_lattice
from server.galaxy import Row
from server.responses import BINARY_OCTET, RevalidatedRoute

router = APIRouter(tags=["encoding"], route_class=RevalidatedRoute)


class Galaxy(BaseModel):
    object_id: str
    ra: float
    dec: float
    ls: bool
    hsc: bool
    desi: bool
    sdss: bool


@router.get(
    "/galaxy/{galaxy}",
)
def get_galaxy(row: Row) -> Galaxy:
    table = tokens().slice(row, 1)
    return Galaxy(
        **identity(row),
        **{survey: table.column(survey)[0].is_valid for survey in TOKEN_SURVEYS},
    )


@router.get(
    "/galaxy/{galaxy}/image/{survey}/false_color",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_image_false_color(galaxy: str, row: Row, survey: ImageSurvey) -> Response:
    cell = tokens().column(survey)[row]
    if not cell.is_valid:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} image")
    codes = np.asarray(cell.values)[: LegacySurveyImage.num_tokens]
    return Response(
        false_colors(image_lattice()[codes]).tobytes(),
        media_type="application/octet-stream",
    )


@router.get(
    "/galaxy/{galaxy}/spectrum/{survey}/false_color",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_spectrum_false_color(galaxy: str, row: Row, survey: SpectrumSurvey) -> Response:
    cell = tokens().column(survey)[row]
    if not cell.is_valid:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} spectrum")
    codes = np.asarray(cell.values)[1:]
    return Response(
        false_colors(spectrum_lattice()[codes]).tobytes(),
        media_type="application/octet-stream",
    )
