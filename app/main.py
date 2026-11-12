from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import pyarrow as pa
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import LegacySurveyImage
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRoute
from fastapi.security import APIKeyCookie, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from . import pql
from .config import (
    AION_SPECTRUM_GRID,
    AION_SPECTRUM_ORIGIN,
    AION_SPECTRUM_TOKEN_WIDTH,
    DATASET_AUTHOR,
    DATASET_NAME,
    FLAG_SURVEYS,
    GRAM_NAMES,
    IMAGE_MODES,
    MODES,
    OBSERVATIONS,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    REDSHIFT_LIMIT,
    REDSHIFT_TABLE_VALUE,
    SPECTRUM_MODES,
    SPECTRUM_SURVEY,
    TABLE_MODES,
    TABLE_VALUE_COLUMNS,
    TOKEN_SURVEYS,
    Catalogue,
    GalaxyIndex,
    ImageSurvey,
    ModeName,
    Projection,
    SpectrumSurvey,
    TokenMode,
    artifact,
    galaxy_count,
)
from .dataset import (
    catalogue,
    fitted,
    identity,
    image,
    positions,
    predicted_image,
    redshift,
    smoothed,
    spectrum,
    table,
)
from .search import (
    FIRST_LS_TABLE_VALUE,
    aion_predictions,
    table_value_aion_predictions,
    table_value_tokens,
    tokens,
)
from .text_search import TextQuery, embed_queries, text_search

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine
    from pathlib import Path


class Meta(BaseModel):
    author: str
    id: str
    galaxy_count: int
    grid: int
    spectrum_origin: float
    spectrum_width: float


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


class TableRow(BaseModel):
    section: Catalogue | Literal["redshift"]
    column: str
    value: float | int | bool | None
    aion_prediction: float | None
    table_value: int | None
    token: int | None
    excluded: str | None


class TextMatches(BaseModel):
    galaxies: list[int]
    scores: list[float]
    contributions: dict[ModeName, list[float]]


BINARY_OCTET: dict[str, Any] = {
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
}
ARROW_STREAM: dict[str, Any] = {
    "application/vnd.apache.arrow.stream": {
        "schema": {"type": "string", "format": "binary"}
    }
}
PNG: dict[str, Any] = {"image/png": {"schema": {"type": "string", "format": "binary"}}}


def arrow(data: pa.RecordBatch | pa.Table) -> Response:
    sink = BytesIO()
    with pa.ipc.new_stream(sink, data.schema) as writer:
        writer.write(data)
    return Response(sink.getvalue(), media_type="application/vnd.apache.arrow.stream")


def not_modified(request: Request, etag: str) -> bool:
    tags = request.headers.get("if-none-match", "")
    return etag in [tag.strip().removeprefix("W/") for tag in tags.split(",")]


class RevalidatedRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def with_cache_headers(request: Request) -> Response:
            response = await handler(request)
            if response.status_code == 200 and "etag" not in response.headers:
                etag = f'"{hashlib.md5(response.body).hexdigest()}"'
                if not_modified(request, etag):
                    response = Response(status_code=304, headers={"etag": etag})
                else:
                    response.headers["etag"] = etag
            response.headers["cache-control"] = "no-cache"
            return response

        return with_cache_headers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    galaxy_count()
    pql.predictions()
    pql.basis()
    for column in OBSERVATIONS:
        pql.observed(column)
    for mode in MODES:
        aion_predictions(mode)
    for name in GRAM_NAMES:
        pql.stored(name)
    yield


PASSWORD_COOKIE = "password"
PASSWORD_COOKIE_AGE = 30 * 24 * 60 * 60

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


app = FastAPI(
    title="alphauniverse",
    lifespan=lifespan,
    generate_unique_id_function=lambda route: route.name,
    dependencies=[Depends(require_password)],
)
app.router.route_class = RevalidatedRoute


@app.post("/login")
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


@app.get(
    "/meta",
)
def get_meta() -> Meta:
    return Meta(
        author=DATASET_AUTHOR,
        id=DATASET_NAME,
        galaxy_count=galaxy_count(),
        grid=MODALITY_INFO["tok_image"]["input_size"]
        // MODALITY_INFO["tok_image"]["patch_size"],
        spectrum_origin=AION_SPECTRUM_ORIGIN,
        spectrum_width=AION_SPECTRUM_TOKEN_WIDTH,
    )


def file(path: Path, request: Request) -> Response:
    try:
        stat_result = path.stat()
    except OSError as exception:
        raise HTTPException(404, str(exception)) from exception

    response = FileResponse(
        path,
        media_type="application/octet-stream",
        filename=path.name,
        stat_result=stat_result,
    )
    if not_modified(request, response.headers["etag"]):
        return Response(status_code=304, headers={"etag": response.headers["etag"]})
    return response


@app.get(
    "/projections/{projection}",
    response_class=FileResponse,
    responses={200: {"content": BINARY_OCTET}},
)
def get_projections(projection: Projection, request: Request) -> Response:
    return file(artifact(f"{projection}_points"), request)


@app.head("/projections/{projection}", include_in_schema=False)
def head_projections(projection: Projection, request: Request) -> Response:
    return get_projections(projection, request)


@app.get(
    "/galaxies/property",
    response_class=FileResponse,
    responses={200: {"content": BINARY_OCTET}},
)
def get_properties(request: Request) -> Response:
    return file(artifact("properties"), request)


@app.head("/galaxies/property", include_in_schema=False)
def head_properties(request: Request) -> Response:
    return get_properties(request)


@app.get(
    "/positions",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_positions() -> Response:
    return arrow(positions())


@app.get(
    "/patterns",
    response_class=FileResponse,
    responses={200: {"content": BINARY_OCTET}},
)
def get_patterns(request: Request) -> Response:
    return file(artifact("patterns"), request)


@app.get(
    "/galaxy/{galaxy}",
)
def get_galaxy(galaxy: GalaxyIndex) -> Galaxy:
    table = tokens().slice(galaxy, 1)
    return Galaxy(
        **identity(galaxy),
        **{survey: table.column(survey)[0].is_valid for survey in TOKEN_SURVEYS},
        **{survey: bool(table.column(survey)[0].as_py()) for survey in FLAG_SURVEYS},
    )


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


def catalogue_row(
    column: str, value: Any, token_ids: dict[str, int], predicted: dict[str, float]
) -> TableRow:
    table_value = (
        FIRST_LS_TABLE_VALUE + TABLE_VALUE_COLUMNS.index(column)
        if column in TABLE_VALUE_COLUMNS
        else None
    )
    return TableRow(
        section=catalogue(column),
        column=column.partition("-")[0],
        value=value,
        aion_prediction=predicted.get(column),
        table_value=table_value,
        token=None if table_value is None else token_ids.get(column),
        excluded=None,
    )


@app.get(
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
        catalogue_row(column, value, token_ids, predicted)
        for column, value in values.items()
        if column not in redshifts
    ]


@app.get(
    "/galaxy/{galaxy}/image/{survey}",
    response_class=Response,
    responses={200: {"content": PNG}},
)
def get_image(galaxy: GalaxyIndex, survey: ImageSurvey) -> Response:
    found = image(galaxy, survey)
    if found is None:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} image")
    return Response(found, media_type="image/png")


@app.get(
    "/galaxy/{galaxy}/image/{survey}/tokens",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_image_tokens(galaxy: GalaxyIndex, survey: ImageSurvey) -> Response:
    cell = tokens().column(survey)[galaxy]
    if not cell.is_valid:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} image")
    return Response(
        np.asarray(cell.values)[: LegacySurveyImage.num_tokens].tobytes(),
        media_type="application/octet-stream",
    )


@app.get(
    "/galaxy/{galaxy}/image/{survey}/aion_prediction",
    response_class=Response,
    responses={200: {"content": PNG}},
)
def get_image_aion_prediction(galaxy: GalaxyIndex, survey: ImageSurvey) -> Response:
    return Response(
        predicted_image(aion_predictions(f"{survey}_image")[galaxy], survey),
        media_type="image/png",
    )


@app.get(
    "/galaxy/{galaxy}/spectrum/{survey}",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_spectrum(galaxy: GalaxyIndex, survey: SpectrumSurvey) -> Response:
    table = spectrum(galaxy, survey)
    if table is None:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} spectrum")
    flux = smoothed(table["flux"].to_numpy())
    return arrow(table.set_column(1, "flux", pa.array(flux, pa.float32())))


@app.get(
    "/galaxy/{galaxy}/spectrum/{survey}/tokens",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_spectrum_tokens(galaxy: GalaxyIndex, survey: SpectrumSurvey) -> Response:
    cell = tokens().column(survey)[galaxy]
    if not cell.is_valid:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} spectrum")
    return Response(
        np.asarray(cell.values)[1:].tobytes(), media_type="application/octet-stream"
    )


@app.get(
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


@app.get(
    "/galaxy/{galaxy}/cosines/{mode}",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_cosines(galaxy: GalaxyIndex, mode: TokenMode) -> Response:
    return Response(
        pql.cosines(galaxy, mode).tobytes(), media_type="application/octet-stream"
    )


def map_arrays(maps: dict[str, np.ndarray]) -> tuple[list[pa.Field], list[pa.Array]]:
    item = pa.field("item", pa.float32(), nullable=False)
    return (
        [
            pa.field(mode, pa.list_(item, values.shape[1]), nullable=False)
            for mode, values in maps.items()
        ],
        [
            pa.FixedSizeListArray.from_arrays(
                pa.array(values.reshape(-1), pa.float32()), values.shape[1]
            )
            for values in maps.values()
        ],
    )


def search_response(
    fields: list[pa.Field],
    columns: list[pa.Array],
    galaxies: np.ndarray,
    predicted: list[list[str]],
) -> Response:
    observations = pa.list_(pa.field("item", pa.string(), nullable=False))
    fields = [
        *fields,
        *(
            pa.field(f"has_{column}", pa.bool_(), nullable=False)
            for column in OBSERVATIONS
        ),
        pa.field("predicted", observations, nullable=False),
    ]
    columns = [
        *columns,
        *(pa.array(pql.observed(column)[galaxies]) for column in OBSERVATIONS),
        pa.array(predicted, observations),
    ]
    return arrow(pa.record_batch(columns, schema=pa.schema(fields)))


@app.get(
    "/saliency",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_saliency(query: Annotated[pql.Selection, Query()]) -> Response:
    fields, columns = map_arrays(pql.by_mode(pql.saliency(query)[None]))
    return arrow(pa.record_batch(columns, schema=pa.schema(fields)))


@app.get(
    "/search",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_search(query: Annotated[pql.Query, Query()]) -> Response:
    results = pql.search(query)
    count = len(results.galaxies)
    fields = [
        pa.field("galaxy", pa.int32(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
        pa.field("similarity", pa.float32(), nullable=False),
    ]
    columns = [
        pa.array(results.galaxies),
        pa.array(results.scores),
        pa.array(results.similarity, pa.float32()),
    ]
    map_fields, map_columns = map_arrays(results.maps)
    fields += map_fields
    columns += map_columns
    for mode in (*IMAGE_MODES, *SPECTRUM_MODES, *TABLE_MODES):
        fields.append(pa.field(f"{mode}_sum", pa.float32()))
        sums = results.sums.get(mode)
        columns.append(
            pa.nulls(count, pa.float32())
            if sums is None
            else pa.array(sums, pa.float32())
        )
        fields.append(pa.field(f"{mode}_similarity", pa.float32()))
        shares = results.similarities.get(mode)
        columns.append(
            pa.nulls(count, pa.float32())
            if shares is None
            else pa.array(shares, pa.float32())
        )
    places = pa.list_(pa.field("item", pa.int32(), nullable=False))
    for mode in (*IMAGE_MODES, *SPECTRUM_MODES):
        fields.append(pa.field(f"{mode}_positions", places))
        positions = results.positions.get(mode)
        columns.append(
            pa.nulls(count, places)
            if positions is None
            else pa.array(positions.tolist(), places)
        )
    return search_response(fields, columns, results.galaxies, results.predicted)


@app.get(
    "/search/modes",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_mode_search(query: Annotated[pql.ModeQuery, Query()]) -> Response:
    results = pql.mode_search(query)
    count = len(results.galaxies)
    fields = [
        pa.field("galaxy", pa.int32(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
    ]
    columns = [pa.array(results.galaxies), pa.array(results.scores)]
    map_fields, map_columns = map_arrays(results.maps)
    fields += map_fields
    columns += map_columns
    for mode in MODES:
        fields.append(pa.field(f"{mode}_contribution", pa.float32()))
        values = results.contributions.get(mode)
        columns.append(pa.nulls(count) if values is None else pa.array(values))
    return search_response(fields, columns, results.galaxies, results.predicted)


def query_embedder() -> Callable[[list[str]], np.ndarray]:
    return embed_queries


@app.get("/search/text")
def get_text_search(
    query: Annotated[TextQuery, Query()],
    embed: Annotated[Callable[[list[str]], np.ndarray], Depends(query_embedder)],
) -> TextMatches:
    found = text_search(query, embed)
    return TextMatches(
        galaxies=found.galaxies.tolist(),
        scores=found.scores.tolist(),
        contributions=dict(zip(MODES, found.contributions.T.tolist())),
    )
