import numpy as np
from aion.modalities import LegacySurveyImage
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import (
    FLAG_SURVEYS,
    TOKEN_SURVEYS,
    GalaxyIndex,
    ImageSurvey,
    SpectrumSurvey,
)
from app.dataset.dataset import identity
from app.encoding.store import false_colors, image_lattice, spectrum_lattice, tokens
from app.responses import BINARY_OCTET, RevalidatedRoute

router = APIRouter(tags=["encoding"], route_class=RevalidatedRoute)


class Galaxy(BaseModel):
    object_id: str
    ra: float
    dec: float
    ls: bool
    hsc: bool
    desi: bool
    sdss: bool
    gz10: bool
    provabgs: bool


@router.get(
    "/galaxy/{galaxy}",
)
def get_galaxy(galaxy: GalaxyIndex) -> Galaxy:
    table = tokens().slice(galaxy, 1)
    return Galaxy(
        **identity(galaxy),
        **{survey: table.column(survey)[0].is_valid for survey in TOKEN_SURVEYS},
        **{survey: bool(table.column(survey)[0].as_py()) for survey in FLAG_SURVEYS},
    )


@router.get(
    "/galaxy/{galaxy}/image/{survey}/false_color",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_image_false_color(galaxy: GalaxyIndex, survey: ImageSurvey) -> Response:
    cell = tokens().column(survey)[galaxy]
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
def get_spectrum_false_color(galaxy: GalaxyIndex, survey: SpectrumSurvey) -> Response:
    cell = tokens().column(survey)[galaxy]
    if not cell.is_valid:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} spectrum")
    codes = np.asarray(cell.values)[1:]
    return Response(
        false_colors(spectrum_lattice()[codes]).tobytes(),
        media_type="application/octet-stream",
    )
