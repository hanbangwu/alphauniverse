import pyarrow as pa
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from alphauniverse.config import ImageSurvey, SpectrumSurvey
from alphauniverse.dataset.dataset import image, smoothed, spectrum
from server.dataset.dataset import positions
from server.galaxy import Row
from server.responses import ARROW_STREAM, PNG, RevalidatedRoute, arrow

router = APIRouter(tags=["dataset"], route_class=RevalidatedRoute)


@router.get(
    "/positions",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_positions() -> Response:
    return arrow(positions())


@router.get(
    "/galaxy/{galaxy}/image/{survey}",
    response_class=Response,
    responses={200: {"content": PNG}},
)
def get_image(galaxy: str, row: Row, survey: ImageSurvey) -> Response:
    found = image(row, survey)
    if found is None:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} image")
    return Response(found, media_type="image/png")


@router.get(
    "/galaxy/{galaxy}/spectrum/{survey}",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_spectrum(galaxy: str, row: Row, survey: SpectrumSurvey) -> Response:
    table = spectrum(row, survey)
    if table is None:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} spectrum")
    flux = smoothed(table["flux"].to_numpy())
    return arrow(table.set_column(1, "flux", pa.array(flux, pa.float32())))
