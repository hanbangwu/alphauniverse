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
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

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
)
from .cutouts import cutout, cutouts
from .search import Query as SearchQuery
from .search import index, search, source, starts, with_spectrum
from .spectra import spectra, spectrum

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


@cache
def labels() -> tuple[int, list[int], int]:
    points = pq.read_table(artifact("mean_points"), columns=["galaxy", "category"])
    category = points["category"]
    labelled = np.asarray(category.drop_null())
    return (
        len(category),
        np.bincount(labelled, minlength=N_MORPHOLOGIES).tolist(),
        category.null_count,
    )


class SpectrumGrid(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Where spectral token `i` sits, in Ångström.\n\n"
                "From `origin + i * width` to `origin + (i + 1) * width`."
            )
        }
    )

    origin: float
    width: float


class Meta(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "What the frontend needs before it can render anything."
        }
    )

    author: str
    id: str
    revision: str
    galaxies: int
    grid: int
    spectrum: SpectrumGrid
    embeddings: str
    mean_points: str
    full_points: str
    morphologies: list[int]
    unlabelled: int


class Survey(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "Whether one galaxy was crossmatched into one survey."
        }
    )

    survey: str
    matched: bool


class Detail(BaseModel):
    model_config = ConfigDict(json_schema_extra={"description": "An error body."})

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    galaxies = labels()[0]
    for role, load in (("cutouts", cutouts), ("spectra", spectra)):
        stored = len(load())
        if stored != galaxies:
            raise ValueError(f"{stored} {role} for {galaxies} galaxies")

    index()
    starts()
    yield


app = FastAPI(
    title="alphauniverse",
    lifespan=lifespan,
    generate_unique_id_function=lambda route: route.name,
)


@app.middleware("http")
async def cache_headers(
    request: Request, call_next: Callable[[Request], Awaitable[StreamingResponse]]
) -> Response:
    response = await call_next(request)
    if response.status_code == 200 and "etag" not in response.headers:
        body = b"".join([chunk async for chunk in response.body_iterator])
        etag = f'"{hashlib.md5(body, usedforsecurity=False).hexdigest()}"'
        if not_modified(request, etag):
            return Response(status_code=304, headers={"etag": etag})
        response = Response(body, response.status_code, response.headers)
        response.headers["etag"] = etag
    return response


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
    description="Dataset identity, size, patch grid, artifact roles and label counts.",
)
def get_meta() -> Meta:
    galaxies, morphologies, unlabelled = labels()
    return Meta(
        author=DATASET_AUTHOR,
        id=DATASET_NAME,
        revision=DATASET_REVISION,
        galaxies=galaxies,
        grid=GRID,
        spectrum=SpectrumGrid(origin=SPECTRUM_ORIGIN, width=SPECTRUM_TOKEN_WIDTH),
        embeddings="encoded",
        mean_points="mean_points",
        full_points="full_points",
        morphologies=morphologies,
        unlabelled=unlabelled,
    )


@app.get(
    "/artifacts/{role}",
    response_class=FileResponse,
    responses={200: {"content": BINARY_OCTET}, **NOT_FOUND},
    description="Download a build artifact by role.",
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
    "/galaxies/{galaxy}/image.png",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}
        }
    },
    description="The galaxy's anchor-survey cutout as a PNG.",
)
def get_image(galaxy: GalaxyIndex) -> Response:
    return Response(cutout(galaxy), media_type="image/png")


@app.get(
    "/galaxies/{galaxy}/tokens",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
    description="The galaxy's anchor image token ids, as raw little-endian uint32.",
)
def get_tokens(galaxy: GalaxyIndex) -> Response:
    table = source("tokens").to_table(
        columns=[ANCHOR], filter=ds.field("galaxy") == galaxy
    )
    cell = table.column(ANCHOR).combine_chunks()[0]
    return Response(
        np.asarray(cell.values)[:N_PATCHES].tobytes(),
        media_type="application/octet-stream",
    )


@app.get(
    "/galaxies/{galaxy}/spectra/{survey}",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}, **NOT_FOUND},
    description="The galaxy's spectrum from one survey, as Arrow IPC.",
)
def get_spectrum(galaxy: GalaxyIndex, survey: SpectrumSurvey) -> Response:
    table = spectrum(galaxy, survey)
    if table is None:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} spectrum")
    return arrow(table)


@app.get(
    "/galaxies/{galaxy}/spectra/{survey}/tokens",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}, **NOT_FOUND},
    description="The galaxy's spectrum token ids from one survey, as raw uint32.",
)
def get_spectrum_tokens(galaxy: GalaxyIndex, survey: SpectrumSurvey) -> Response:
    table = source("tokens").to_table(
        columns=[survey], filter=ds.field("galaxy") == galaxy
    )
    cell = table.column(survey)[0]
    if not cell.is_valid:
        raise HTTPException(404, f"galaxy {galaxy} has no {survey} spectrum")
    return Response(
        np.asarray(cell.values)[1:].tobytes(), media_type="application/octet-stream"
    )


@app.get(
    "/galaxies/{galaxy}/coverage",
    description="Which surveys the galaxy was crossmatched into.",
)
def get_coverage(galaxy: GalaxyIndex) -> list[Survey]:
    table = source("tokens").to_table(
        columns=[*TOKEN_SURVEYS, *FLAG_SURVEYS], filter=ds.field("galaxy") == galaxy
    )
    return [
        Survey(survey=survey, matched=table.column(survey)[0].is_valid)
        for survey in TOKEN_SURVEYS
    ] + [
        Survey(survey=survey, matched=bool(table.column(survey)[0].as_py()))
        for survey in FLAG_SURVEYS
    ]


@app.get(
    "/similarity",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
    description=(
        "Galaxies ranked against the mean of the query tokens, as Arrow IPC.\n\n"
        "One record batch of four columns: the galaxy index, its best token score,\n"
        "a fixed-size list of one score per image patch, and one per spectral span\n"
        "or null for a galaxy without a spectrum. Row 0 is always the query galaxy\n"
        "itself."
    ),
)
def get_similarity(query: Annotated[SearchQuery, Query()]) -> Response:
    if query.spans and not with_spectrum()[query.galaxy]:
        raise HTTPException(422, f"galaxy {query.galaxy} has no spectrum")
    galaxies, scores, values, spans = search(query, index=index())

    item = pa.field("item", pa.float32(), nullable=False)
    schema = pa.schema(
        [
            pa.field("galaxy", pa.int32(), nullable=False),
            pa.field("score", pa.float32(), nullable=False),
            pa.field("map", pa.list_(item, N_PATCHES), nullable=False),
            pa.field("spectrum", pa.list_(item, N_SPANS)),
        ],
        metadata={
            "revision": DATASET_REVISION,
            "galaxy": str(query.galaxy),
            "patches": ",".join(map(str, query.patches)),
            "spans": ",".join(map(str, query.spans)),
            "n_patches": str(N_PATCHES),
            "n_spans": str(N_SPANS),
        },
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
