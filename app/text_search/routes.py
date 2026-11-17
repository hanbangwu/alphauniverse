from collections.abc import Callable
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import MODES, ModeName
from app.responses import ECSV, RevalidatedRoute
from app.text_search.export import text_search_ecsv
from app.text_search.text_search import TextQuery, embed_queries, text_search

router = APIRouter(tags=["text search"], route_class=RevalidatedRoute)


class TextMatches(BaseModel):
    galaxies: list[int]
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
        galaxies=found.galaxies.tolist(),
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
