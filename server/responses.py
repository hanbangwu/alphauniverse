from __future__ import annotations

import hashlib
from io import BytesIO
from typing import TYPE_CHECKING, Any

import pyarrow as pa
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.routing import APIRoute

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

BINARY_OCTET: dict[str, Any] = {
    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
}
ARROW_STREAM: dict[str, Any] = {
    "application/vnd.apache.arrow.stream": {
        "schema": {"type": "string", "format": "binary"}
    }
}
PNG: dict[str, Any] = {"image/png": {"schema": {"type": "string", "format": "binary"}}}
ECSV: dict[str, Any] = {"text/plain": {"schema": {"type": "string"}}}


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
