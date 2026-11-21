from typing import Any, Literal

import pyarrow as pa
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import (
    AION_SPECTRUM_GRID,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    REDSHIFT_LIMIT,
    REDSHIFT_TABLE_VALUE,
    SPECTRUM_SURVEY,
    TABLE_VALUE_COLUMNS,
    Catalog,
    GalaxyIndex,
    ImageSurvey,
    SpectrumSurvey,
)
from app.dataset.dataset import catalog, fitted, predicted_image, redshift, table
from app.encoding.store import (
    FIRST_LS_TABLE_VALUE,
    false_colors,
    image_lattice,
    observed,
    spectrum_lattice,
)
from app.predictions.store import (
    aion_predictions,
    dense_image_tokens,
    distributions,
    row,
    spectrum_token_coefficients,
    table_value_aion_predictions,
)
from app.responses import ARROW_STREAM, BINARY_OCTET, PNG, RevalidatedRoute, arrow

router = APIRouter(tags=["predictions"], route_class=RevalidatedRoute)


class TableRow(BaseModel):
    section: Catalog | Literal["redshift"]
    column: str
    value: float | int | bool | None
    aion_prediction: float | None
    table_value: int | None
    excluded: str | None


def excluded(values: dict, survey: SpectrumSurvey, chosen: str | None) -> str | None:
    value, _, warning = (values[column] for column in REDSHIFT_COLUMNS[survey])
    if value is None or survey == chosen:
        return None
    if warning:
        return f"flagged by {survey.upper()}"
    if not value <= REDSHIFT_LIMIT:
        return f"above AION's redshift limit, {REDSHIFT_LIMIT:g}"
    return f"AION takes one redshift: {chosen.upper()}'s"


def redshift_row(
    column: str,
    survey: SpectrumSurvey,
    values: dict,
    chosen: str | None,
    predicted: dict[str, float],
) -> TableRow:
    first = column == REDSHIFT_COLUMNS[survey][0]
    selectable = first and survey == chosen
    return TableRow(
        section=REDSHIFT,
        column=f"{survey.upper()} {column.partition('-')[0]}",
        value=values[column],
        aion_prediction=predicted[REDSHIFT]
        if first and survey == (chosen or SPECTRUM_SURVEY)
        else None,
        table_value=REDSHIFT_TABLE_VALUE if selectable else None,
        excluded=excluded(values, survey, chosen) if first else None,
    )


def catalog_row(
    galaxy: int, column: str, value: Any, predicted: dict[str, float]
) -> TableRow:
    section = catalog(column)
    table_value = (
        FIRST_LS_TABLE_VALUE + TABLE_VALUE_COLUMNS.index(column)
        if column in TABLE_VALUE_COLUMNS and observed(section)[galaxy]
        else None
    )
    return TableRow(
        section=section,
        column=column.partition("-")[0],
        value=value,
        aion_prediction=predicted.get(column),
        table_value=table_value,
        excluded=None,
    )


@router.get(
    "/galaxy/{galaxy}/table",
)
def get_table(galaxy: GalaxyIndex) -> list[TableRow]:
    predicted = table_value_aion_predictions(galaxy)
    values = table(galaxy)
    chosen = redshift(values)
    shown = [
        (column, survey)
        for survey, columns in REDSHIFT_COLUMNS.items()
        for column in columns
    ]
    redshifts = {column for column, _ in shown}
    return [
        redshift_row(column, survey, values, chosen, predicted)
        for column, survey in shown
    ] + [
        catalog_row(galaxy, column, value, predicted)
        for column, value in values.items()
        if column not in redshifts
    ]


@router.get(
    "/galaxy/{galaxy}/image/{survey}/aion_prediction",
    response_class=Response,
    responses={200: {"content": PNG}},
)
def get_image_aion_prediction(galaxy: GalaxyIndex, survey: ImageSurvey) -> Response:
    return Response(
        predicted_image(aion_predictions(f"{survey}_image")[galaxy], survey),
        media_type="image/png",
    )


@router.get(
    "/galaxy/{galaxy}/image/{survey}/aion_prediction/false_color",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_image_aion_prediction_false_color(
    galaxy: GalaxyIndex, survey: ImageSurvey
) -> Response:
    dense = dense_image_tokens(row(galaxy), survey, slice(None))[0]
    return Response(
        false_colors(dense @ image_lattice()).tobytes(),
        media_type="application/octet-stream",
    )


@router.get(
    "/galaxy/{galaxy}/spectrum/{survey}/aion_prediction",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_spectrum_aion_prediction(
    galaxy: GalaxyIndex, survey: SpectrumSurvey
) -> Response:
    flux = fitted(aion_predictions(f"{survey}_spectrum")[galaxy], galaxy, survey)
    return arrow(
        pa.table(
            {
                "wavelength": pa.array(AION_SPECTRUM_GRID, pa.float32()),
                "flux": pa.array(flux, pa.float32()),
            }
        )
    )


@router.get(
    "/galaxy/{galaxy}/spectrum/{survey}/aion_prediction/false_color",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_spectrum_aion_prediction_false_color(
    galaxy: GalaxyIndex, survey: SpectrumSurvey
) -> Response:
    coefficients = spectrum_token_coefficients(row(galaxy), survey, slice(None))[0]
    dense = distributions(f"{survey}_spectrum", coefficients)
    return Response(
        false_colors(dense @ spectrum_lattice()).tobytes(),
        media_type="application/octet-stream",
    )
