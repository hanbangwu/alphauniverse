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
from app.encoding.store import FIRST_LS_TABLE_VALUE, table_value_tokens
from app.predictions.store import aion_predictions, table_value_aion_predictions
from app.responses import ARROW_STREAM, PNG, RevalidatedRoute, arrow

router = APIRouter(tags=["predictions"], route_class=RevalidatedRoute)


class TableRow(BaseModel):
    section: Catalog | Literal["redshift"]
    column: str
    value: float | int | bool | None
    aion_prediction: float | None
    table_value: int | None
    token: int | None
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
    token_ids: dict[str, int],
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
        token=token_ids.get(REDSHIFT) if selectable else None,
        excluded=excluded(values, survey, chosen) if first else None,
    )


def catalog_row(
    column: str, value: Any, token_ids: dict[str, int], predicted: dict[str, float]
) -> TableRow:
    table_value = (
        FIRST_LS_TABLE_VALUE + TABLE_VALUE_COLUMNS.index(column)
        if column in TABLE_VALUE_COLUMNS
        else None
    )
    return TableRow(
        section=catalog(column),
        column=column.partition("-")[0],
        value=value,
        aion_prediction=predicted.get(column),
        table_value=table_value,
        token=None if table_value is None else token_ids.get(column),
        excluded=None,
    )


@router.get(
    "/galaxy/{galaxy}/table",
)
def get_table(galaxy: GalaxyIndex) -> list[TableRow]:
    token_ids = table_value_tokens(galaxy)
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
        redshift_row(column, survey, values, chosen, token_ids, predicted)
        for column, survey in shown
    ] + [
        catalog_row(column, value, token_ids, predicted)
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
