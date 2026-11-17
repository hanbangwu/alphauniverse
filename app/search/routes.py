from typing import Annotated

import numpy as np
import pyarrow as pa
from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.config import (
    IMAGE_MODES,
    MODES,
    OBSERVATIONS,
    SPECTRUM_MODES,
    TABLE_MODES,
    GalaxyIndex,
    TokenMode,
)
from app.encoding.store import observed
from app.responses import ARROW_STREAM, BINARY_OCTET, ECSV, RevalidatedRoute, arrow
from app.search import pql
from app.search.export import search_ecsv

router = APIRouter(tags=["search"], route_class=RevalidatedRoute)


@router.get(
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
        *(pa.array(observed(column)[galaxies]) for column in OBSERVATIONS),
        pa.array(
            [[OBSERVATIONS[column] for column in row] for row in predicted],
            observations,
        ),
    ]
    return arrow(pa.record_batch(columns, schema=pa.schema(fields)))


@router.get(
    "/saliency",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_saliency(query: Annotated[pql.Selection, Query()]) -> Response:
    fields, columns = map_arrays(pql.by_mode(pql.saliency(query)[None]))
    return arrow(pa.record_batch(columns, schema=pa.schema(fields)))


@router.get(
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


@router.get(
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


@router.get(
    "/search/download",
    response_class=Response,
    responses={200: {"content": ECSV}},
)
def get_search_download(query: Annotated[pql.Query, Query()]) -> Response:
    results = pql.search(query)
    return Response(
        search_ecsv(results.galaxies, results.similarity, results.maps, query),
        media_type="text/plain",
    )


@router.get(
    "/search/modes/download",
    response_class=Response,
    responses={200: {"content": ECSV}},
)
def get_mode_search_download(query: Annotated[pql.ModeQuery, Query()]) -> Response:
    results = pql.mode_search(query)
    return Response(
        search_ecsv(results.galaxies, results.scores, results.maps, query),
        media_type="text/plain",
    )
