from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from functools import cache
from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel

from .config import (
    ANCHOR,
    DATASET_AUTHOR,
    DATASET_NAME,
    DATASET_REVISION,
    FLAG_SURVEYS,
    GRID,
    N_MORPHOLOGIES,
    N_PATCHES,
    N_SPANS,
    SPECTRUM_ORIGIN,
    SPECTRUM_TOKEN_WIDTH,
    TOKEN_SURVEYS,
    GalaxyIndex,
    SpectrumSurvey,
    artifact,
    galaxy_count,
)
from .images import image, images
from .search import Query as SearchQuery
from .search import index, search, source, starts
from .spectra import spectra, spectrum

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

SPECTRUM_SURVEY: SpectrumSurvey = "desi"


@cache
def labels() -> tuple[list[int], int]:
    category = pq.read_table(artifact("mean_points"), columns=["category"])["category"]
    labelled = np.asarray(category.drop_null())
    return (
        np.bincount(labelled, minlength=N_MORPHOLOGIES).tolist(),
        category.null_count,
    )


class SpectrumGrid(BaseModel):
    origin: float
    width: float


class Meta(BaseModel):
    author: str
    id: str
    revision: str
    galaxies: int
    grid: int
    spectrum: SpectrumGrid
    morphologies: list[int]
    unlabelled: int


class Survey(BaseModel):
    survey: str
    matched: bool


class Galaxy(BaseModel):
    coverage: list[Survey]


class Detail(BaseModel):
    detail: str


NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": Detail}}
BINARY_OCTET: dict[str, Any] = {
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
}
ARROW_STREAM: dict[str, Any] = {
    "application/vnd.apache.arrow.stream": {
        "schema": {"type": "string", "format": "binary"}
    }
}


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
    labels()
    images()
    spectra()
    index()
    starts()
    yield


app = FastAPI(
    title="alphauniverse",
    lifespan=lifespan,
    generate_unique_id_function=lambda route: route.name,
)
app.router.route_class = RevalidatedRoute

app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        "https://alphauniverse.aiforscientists.org",
        "https://alphauniverse-surp.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ),
    allow_origin_regex=r"https://alphauniverse[a-z0-9-]*\.vercel\.app",
    allow_methods=["GET", "HEAD"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"],
    max_age=86400,
)


@app.get(
    "/meta",
)
def get_meta() -> Meta:
    morphologies, unlabelled = labels()
    return Meta(
        author=DATASET_AUTHOR,
        id=DATASET_NAME,
        revision=DATASET_REVISION,
        galaxies=galaxy_count(),
        grid=GRID,
        spectrum=SpectrumGrid(origin=SPECTRUM_ORIGIN, width=SPECTRUM_TOKEN_WIDTH),
        morphologies=morphologies,
        unlabelled=unlabelled,
    )


@app.get(
    "/artifacts/{role}",
    response_class=FileResponse,
    responses={200: {"content": BINARY_OCTET}, **NOT_FOUND},
)
def get_artifact(role: str, request: Request) -> Response:
    try:
        path = artifact(role)
        stat_result = path.stat()
    except (KeyError, OSError) as exception:
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


@app.head("/artifacts/{role}", include_in_schema=False)
def head_artifact(role: str, request: Request) -> Response:
    return get_artifact(role, request)


@app.get(
    "/galaxy/{galaxy}/image",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}
        }
    },
)
def get_image(galaxy: GalaxyIndex) -> Response:
    return Response(image(galaxy), media_type="image/png")


@app.get(
    "/galaxy/{galaxy}/image/tokens",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_image_tokens(galaxy: GalaxyIndex) -> Response:
    table = source("tokens").to_table(
        columns=[ANCHOR], filter=ds.field("galaxy") == galaxy
    )
    cell = table.column(ANCHOR).combine_chunks()[0]
    return Response(
        np.asarray(cell.values)[:N_PATCHES].tobytes(),
        media_type="application/octet-stream",
    )


@app.get(
    "/galaxy/{galaxy}/spectrum",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}, **NOT_FOUND},
)
def get_spectrum(galaxy: GalaxyIndex) -> Response:
    table = spectrum(galaxy, SPECTRUM_SURVEY)
    if table is None:
        raise HTTPException(404, f"galaxy {galaxy} has no {SPECTRUM_SURVEY} spectrum")
    return arrow(table)


@app.get(
    "/galaxy/{galaxy}/spectrum/tokens",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}, **NOT_FOUND},
)
def get_spectrum_tokens(galaxy: GalaxyIndex) -> Response:
    table = source("tokens").to_table(
        columns=[SPECTRUM_SURVEY], filter=ds.field("galaxy") == galaxy
    )
    cell = table.column(SPECTRUM_SURVEY)[0]
    if not cell.is_valid:
        raise HTTPException(404, f"galaxy {galaxy} has no {SPECTRUM_SURVEY} spectrum")
    return Response(
        np.asarray(cell.values)[1:].tobytes(), media_type="application/octet-stream"
    )


@app.get(
    "/galaxy/{galaxy}",
)
def get_galaxy(galaxy: GalaxyIndex) -> Galaxy:
    table = source("tokens").to_table(
        columns=[*TOKEN_SURVEYS, *FLAG_SURVEYS], filter=ds.field("galaxy") == galaxy
    )
    return Galaxy(
        coverage=[
            Survey(survey=survey, matched=table.column(survey)[0].is_valid)
            for survey in TOKEN_SURVEYS
        ]
        + [
            Survey(survey=survey, matched=bool(table.column(survey)[0].as_py()))
            for survey in FLAG_SURVEYS
        ]
    )


@app.get(
    "/search",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_search(query: Annotated[SearchQuery, Query()]) -> Response:
    galaxies, scores, values, spans = search(query, index=index())

    item = pa.field("item", pa.float32(), nullable=False)
    schema = pa.schema(
        [
            pa.field("galaxy", pa.int32(), nullable=False),
            pa.field("score", pa.float32(), nullable=False),
            pa.field("map", pa.list_(item, N_PATCHES), nullable=False),
            pa.field("spectrum", pa.list_(item, N_SPANS)),
        ]
    )
    batch = pa.record_batch(
        [
            pa.array(galaxies),
            pa.array(scores),
            pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), N_PATCHES),
            pa.FixedSizeListArray.from_arrays(
                pa.array(spans.reshape(-1)),
                N_SPANS,
                mask=pa.array(np.isnan(spans[:, 0])),
            ),
        ],
        schema=schema,
    )

    return arrow(batch)
