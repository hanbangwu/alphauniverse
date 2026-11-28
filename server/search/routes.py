from typing import Annotated, Any

import numpy as np
import pyarrow as pa
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import ValidationError

import alphauniverse
from alphauniverse.config import (
    IMAGE_MODES,
    MODES,
    OBSERVATIONS,
    REDSHIFT,
    SPECTRUM_MODES,
    TABLE_MODES,
)
from alphauniverse.encoding.store import observed, rows
from alphauniverse.library import INTERNAL_MODES
from alphauniverse.search.pql import (
    OBSERVED_IN,
    TABLE_SLOTS,
)
from server.responses import ARROW_STREAM, ECSV, RevalidatedRoute, arrow
from server.search.export import search_ecsv
from server.search.queries import ModeSearch, Search, Selection

router = APIRouter(tags=["search"], route_class=RevalidatedRoute)

LIBRARY_MODES = {REDSHIFT: "desi_table"}
SHAPES = {
    "ls_image": alphauniverse.IMAGE_TOKEN_GRID,
    "hsc_image": alphauniverse.IMAGE_TOKEN_GRID,
    "desi_spectrum": (alphauniverse.SPECTRUM_TOKENS,),
    "sdss_spectrum": (alphauniverse.SPECTRUM_TOKENS,),
    "ls_table": (len(alphauniverse.LS_TABLE_COLUMNS),),
    "hsc_table": (len(alphauniverse.HSC_TABLE_COLUMNS),),
    "desi_table": (1,),
}


def masks(query: Selection) -> dict[str, np.ndarray]:
    chosen = {
        mode: list(slots)
        for mode in (*IMAGE_MODES, *SPECTRUM_MODES)
        if (slots := getattr(query, mode))
    }
    for index in query.table_values:
        mode, slot = TABLE_SLOTS[index]
        chosen.setdefault(LIBRARY_MODES.get(mode, mode), []).append(slot)
    found = {}
    for name, slots in chosen.items():
        mask = np.zeros(SHAPES[name], bool)
        mask.flat[slots] = True
        found[name] = mask
    return found


def searched(run: Any, *arguments: Any, **options: Any) -> Any:
    try:
        return run(*arguments, **options)
    except (ValueError, ValidationError) as failure:
        raise HTTPException(422, str(failure)) from None


def internal(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {INTERNAL_MODES[name]: value for name, value in values.items()}


def flat(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = len(maps["ls_image"])
    return {
        mode: maps[mode].reshape(count, -1) for mode in (*IMAGE_MODES, *SPECTRUM_MODES)
    } | {
        "table_values": np.concatenate(
            [maps["desi_table"], maps["ls_table"], maps["hsc_table"]], axis=-1
        )
    }


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
    galaxies: list[str],
    predicted: list[list[str]],
) -> Response:
    found = rows(galaxies)
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
        *(pa.array(observed(column)[found]) for column in OBSERVATIONS),
        pa.array(
            [
                list(
                    dict.fromkeys(
                        OBSERVATIONS[OBSERVED_IN[INTERNAL_MODES[name]]] for name in row
                    )
                )
                for row in predicted
            ],
            observations,
        ),
    ]
    return arrow(pa.record_batch(columns, schema=pa.schema(fields)))


@router.get(
    "/saliency",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_saliency(query: Annotated[Selection, Query()]) -> Response:
    found = searched(
        alphauniverse.saliency,
        query.galaxy,
        **masks(query),
        anywhere=query.anywhere,
    )
    fields, columns = map_arrays(
        flat({name: value[None] for name, value in found.items()})
    )
    return arrow(pa.record_batch(columns, schema=pa.schema(fields)))


def patch_search(query: Search) -> alphauniverse.SearchByPatchResults:
    return searched(
        alphauniverse.search_by_patch,
        query.galaxy,
        **masks(query),
        anywhere=query.anywhere,
        observed_only=query.observed_only,
        matches=query.matches,
    )


def mode_search(query: ModeSearch) -> alphauniverse.SearchByModeResults:
    return searched(
        alphauniverse.search_by_mode,
        query.galaxy,
        [LIBRARY_MODES.get(mode, mode) for mode in query.modes],
        anywhere=query.anywhere,
        observed_only=query.observed_only,
        matches=query.matches,
    )


@router.get(
    "/search",
    response_class=Response,
    responses={200: {"content": ARROW_STREAM}},
)
def get_search(query: Annotated[Search, Query()]) -> Response:
    results = patch_search(query)
    count = len(results.galaxies)
    scores = internal(results.modality_scores)
    fields = [
        pa.field("galaxy", pa.string(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
        pa.field("similarity", pa.float32(), nullable=False),
    ]
    columns = [
        pa.array(results.galaxies, pa.string()),
        pa.array(results.scores, pa.float32()),
        pa.array(results.similarity, pa.float32()),
    ]
    map_fields, map_columns = map_arrays(flat(results.maps))
    fields += map_fields
    columns += map_columns
    for mode in (*IMAGE_MODES, *SPECTRUM_MODES, *TABLE_MODES):
        fields.append(pa.field(f"{mode}_sum", pa.float32()))
        sums = scores.get(mode)
        columns.append(
            pa.nulls(count, pa.float32())
            if sums is None
            else pa.array(sums, pa.float32())
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
def get_mode_search(query: Annotated[ModeSearch, Query()]) -> Response:
    results = mode_search(query)
    count = len(results.galaxies)
    contributions = internal(results.modality_scores)
    fields = [
        pa.field("galaxy", pa.string(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
    ]
    columns = [
        pa.array(results.galaxies, pa.string()),
        pa.array(results.scores, pa.float32()),
    ]
    map_fields, map_columns = map_arrays(flat(results.maps))
    fields += map_fields
    columns += map_columns
    for mode in MODES:
        fields.append(pa.field(f"{mode}_contribution", pa.float32()))
        values = contributions.get(mode)
        columns.append(
            pa.nulls(count, pa.float32())
            if values is None
            else pa.array(values, pa.float32())
        )
    return search_response(fields, columns, results.galaxies, results.predicted)


@router.get(
    "/search/download",
    response_class=Response,
    responses={200: {"content": ECSV}},
)
def get_search_download(query: Annotated[Search, Query()]) -> Response:
    results = patch_search(query)
    return Response(
        search_ecsv(results.galaxies, results.similarity, flat(results.maps), query),
        media_type="text/plain",
    )


@router.get(
    "/search/modes/download",
    response_class=Response,
    responses={200: {"content": ECSV}},
)
def get_mode_search_download(query: Annotated[ModeSearch, Query()]) -> Response:
    results = mode_search(query)
    return Response(
        search_ecsv(results.galaxies, results.scores, flat(results.maps), query),
        media_type="text/plain",
    )
