"""FastAPI stuff"""

from __future__ import annotations

import mimetypes
from contextlib import asynccontextmanager
from functools import cache
from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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
    TOKEN_SURVEYS,
    GalaxyIndex,
    artifact,
)
from .dataset import image
from .search import Query as SearchQuery
from .search import index, search, source

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@cache
def labels() -> tuple[int, list[int], int]:
    """The galaxy count, per-morphology counts and unlabelled count."""
    points = pq.read_table(artifact("mean_points"), columns=["galaxy", "category"])
    category = points["category"]
    labelled = np.asarray(category.drop_null())
    return (
        len(category),
        np.bincount(labelled, minlength=N_MORPHOLOGIES).tolist(),
        category.null_count,
    )


class Meta(BaseModel):
    """What the frontend needs before it can render anything."""

    author: str
    id: str
    revision: str
    galaxies: int
    grid: int
    embeddings: str
    mean_points: str
    full_points: str
    morphologies: list[int]
    unlabelled: int


class Survey(BaseModel):
    """Whether one galaxy was crossmatched into one survey."""

    survey: str
    matched: bool


class Detail(BaseModel):
    """An error body."""

    detail: str


NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": Detail}}
BINARY_OCTET: dict[str, Any] = {
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the index and open the token store before serving traffic."""
    index()
    source("tokens")
    yield


app = FastAPI(
    title="alphauniverse",
    lifespan=lifespan,
    generate_unique_id_function=lambda route: route.name,
)

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


@app.get("/meta")
def get_meta() -> Meta:
    """Dataset identity, size, patch grid, artifact roles and label counts."""
    galaxies, morphologies, unlabelled = labels()
    return Meta(
        author=DATASET_AUTHOR,
        id=DATASET_NAME,
        revision=DATASET_REVISION,
        galaxies=galaxies,
        grid=GRID,
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
)
def get_artifact(role: str) -> Response:
    """Download a build artifact by role."""
    try:
        path = artifact(role)
        path.stat()
    except (KeyError, OSError) as exception:
        # The role, not the exception: its message carries the volume path and
        # the revision, and this response goes to anyone who asks.
        raise HTTPException(404, f"no artifact for role {role!r}") from exception

    return FileResponse(
        path,
        media_type=mimetypes.guess_type(path)[0] or "application/octet-stream",
        filename=path.name,
    )


@app.head("/artifacts/{role}", include_in_schema=False)
def head_artifact(role: str) -> Response:
    """Artifact headers only, for range-request clients such as DuckDB."""
    return get_artifact(role)


@app.get(
    "/galaxies/{galaxy}/image.png",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}
        }
    },
)
def get_image(galaxy: GalaxyIndex) -> Response:
    """The galaxy's anchor-survey cutout as a PNG."""
    return Response(image(galaxy), media_type="image/png")


@app.get(
    "/galaxies/{galaxy}/tokens",
    response_class=Response,
    responses={200: {"content": BINARY_OCTET}},
)
def get_tokens(galaxy: GalaxyIndex) -> Response:
    """The galaxy's anchor image token ids, as raw little-endian uint32.

    One value per image patch, in row-major order: see `grid` in /meta.
    """
    table = source("tokens").to_table(
        columns=[ANCHOR], filter=ds.field("galaxy") == galaxy
    )
    cell = table.column(ANCHOR).combine_chunks()[0]
    return Response(
        np.asarray(cell.values)[:N_PATCHES].tobytes(),
        media_type="application/octet-stream",
    )


@app.get("/galaxies/{galaxy}/coverage")
def get_coverage(galaxy: GalaxyIndex) -> list[Survey]:
    """Which surveys the galaxy was crossmatched into."""
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
    responses={
        200: {
            "content": {
                "application/vnd.apache.arrow.stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            }
        }
    },
)
def get_similarity(query: Annotated[SearchQuery, Query()]) -> Response:
    """Galaxies ranked against the mean of the query patches, as Arrow IPC.

    One record batch of three columns: the galaxy index, its best patch score,
    and a fixed-size list of one score per patch. Row 0 is always the query
    galaxy itself. Candidates come from an approximate search, so fewer than
    `matches` rows can come back.
    """
    galaxies, scores, values = search(query, index=index())

    item = pa.field("item", pa.float32(), nullable=False)
    schema = pa.schema(
        [
            pa.field("galaxy", pa.int32(), nullable=False),
            pa.field("score", pa.float32(), nullable=False),
            pa.field("map", pa.list_(item, N_PATCHES), nullable=False),
        ],
        metadata={
            "revision": DATASET_REVISION,
            "galaxy": str(query.galaxy),
            "patches": ",".join(map(str, query.patches)),
            "n_patches": str(N_PATCHES),
        },
    )
    batch = pa.record_batch(
        [
            pa.array(galaxies),
            pa.array(scores),
            pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), N_PATCHES),
        ],
        schema=schema,
    )

    sink = BytesIO()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return Response(sink.getvalue(), media_type="application/vnd.apache.arrow.stream")
