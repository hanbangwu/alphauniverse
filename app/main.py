from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import pyarrow as pa
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel

from .config import (
    ANCHOR,
    DATASET_AUTHOR,
    DATASET_NAME,
    FLAG_SURVEYS,
    GRID,
    N_PATCHES,
    N_SPANS,
    SPECTRUM_ORIGIN,
    SPECTRUM_SURVEY,
    SPECTRUM_TOKEN_WIDTH,
    TOKEN_SURVEYS,
    Download,
    GalaxyIndex,
    Projection,
    artifact,
    galaxy_count,
    labels,
)
from .dataset import image, spectrum
from .search import Query as SearchQuery
from .search import index, search, starts, tokens

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine
    from pathlib import Path


class Meta(BaseModel):
    author: str
    id: str
    galaxies: int
    grid: int
    spectrum_origin: float
    spectrum_width: float
    morphologies: list[int]


class Galaxy(BaseModel):
    ls: bool
    hsc: bool
    desi: bool
    sdss: bool
    gz10: bool
    provabgs: bool


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
    return Meta(
        author=DATASET_AUTHOR,
        id=DATASET_NAME,
        galaxies=galaxy_count(),
        grid=GRID,
        spectrum_origin=SPECTRUM_ORIGIN,
        spectrum_width=SPECTRUM_TOKEN_WIDTH,
        morphologies=labels(),
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
    "/downloads/{role}",
    response_class=FileResponse,
    responses={200: {"content": BINARY_OCTET}},
)
def download_artifact(role: Download, request: Request) -> Response:
    return file(artifact(role), request)


@app.head("/downloads/{role}", include_in_schema=False)
def head_download_artifact(role: Download, request: Request) -> Response:
    return download_artifact(role, request)


@app.get(
    "/galaxy/{galaxy}",
)
def get_galaxy(galaxy: GalaxyIndex) -> Galaxy:
    table = tokens().slice(galaxy, 1)
    return Galaxy(
        **{survey: table.column(survey)[0].is_valid for survey in TOKEN_SURVEYS},
        **{survey: bool(table.column(survey)[0].as_py()) for survey in FLAG_SURVEYS},
    )


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
    cell = tokens().column(ANCHOR)[galaxy]
    return Response(
        np.asarray(cell.values)[:N_PATCHES].tobytes(),
        media_type="application/octet-stream",
    )


@app.get(
    "/galaxy/{galaxy}/spectrum",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_spectrum(galaxy: GalaxyIndex) -> Response:
    table = spectrum(galaxy, SPECTRUM_SURVEY)
    if table is None:
        raise HTTPException(404, f"galaxy {galaxy} has no {SPECTRUM_SURVEY} spectrum")
    return arrow(table)


@app.get(
    "/galaxy/{galaxy}/spectrum/tokens",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_spectrum_tokens(galaxy: GalaxyIndex) -> Response:
    cell = tokens().column(SPECTRUM_SURVEY)[galaxy]
    if not cell.is_valid:
        raise HTTPException(404, f"galaxy {galaxy} has no {SPECTRUM_SURVEY} spectrum")
    return Response(
        np.asarray(cell.values)[1:].tobytes(), media_type="application/octet-stream"
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
