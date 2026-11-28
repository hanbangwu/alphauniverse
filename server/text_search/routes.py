from collections.abc import Callable
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel

from alphauniverse.config import MODES, ModeName
from alphauniverse.encoding.store import object_ids
from alphauniverse.text_search.text_search import TextQuery, embed_queries, text_search
from server.responses import ECSV, RevalidatedRoute
from server.text_search.export import text_search_ecsv

router = APIRouter(tags=["text search"], route_class=RevalidatedRoute)


class TextMatches(BaseModel):
    galaxies: list[str]
    scores: list[float]
    contributions: dict[ModeName, list[float]]


def query_embedder() -> Callable[[list[str]], np.ndarray]:
    return embed_queries


@router.get("/search/text")
def get_text_search(
    query: Annotated[TextQuery, Query()],
    embed: Annotated[Callable[[list[str]], np.ndarray], Depends(query_embedder)],
) -> TextMatches:
    found = text_search(query, embed)
    return TextMatches(
        galaxies=object_ids().take(found.galaxies.numpy(force=True)).to_pylist(),
        scores=found.scores.tolist(),
        contributions=dict(zip(MODES, found.contributions.T.tolist())),
    )


@router.get(
    "/search/text/download",
    response_class=Response,
    responses={200: {"content": ECSV}},
)
def get_text_search_download(
    query: Annotated[TextQuery, Query()],
    embed: Annotated[Callable[[list[str]], np.ndarray], Depends(query_embedder)],
) -> Response:
    found = text_search(query, embed)
    return Response(
        text_search_ecsv(
            found.galaxies.numpy(force=True), found.scores.numpy(force=True), query
        ),
        media_type="text/plain",
    )
