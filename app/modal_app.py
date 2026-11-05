from __future__ import annotations

from typing import TYPE_CHECKING

import modal

from core.modal_resources import (
    CACHE_PATH,
    build_dependencies,
    cache_volume,
    environment,
)

if TYPE_CHECKING:
    import numpy as np
    from fastapi import FastAPI

SERVING_CPU = 8
SERVING_MEMORY = (4 * 1024, 32 * 1024)
SERVING_MAX_INPUTS = 16
SERVING_MAX_CONTAINERS = 1
SERVING_SCALEDOWN_WINDOW = 5 * 60
SERVING_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1"}

app = modal.App("alphauniverse")

query_image = build_dependencies.add_local_python_source("core")


@app.cls(
    image=query_image,
    gpu=["L4", "A10G", "L40S"],
    cpu=2,
    memory=(8 * 1024, 32 * 1024),
    timeout=10 * 60,
    volumes={CACHE_PATH: cache_volume},
    scaledown_window=SERVING_SCALEDOWN_WINDOW,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class QueryEmbedder:
    @modal.enter(snap=True)
    def load(self) -> None:
        from core.text_model import embed_queries

        embed_queries(["a galaxy"])

    @modal.method()
    def embed(self, texts: list[str]) -> np.ndarray:
        from core.text_model import embed_queries

        return embed_queries(texts)


serving_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-dev")
    .env(environment | {"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("app", "core")
)


@app.function(
    image=serving_image,
    env=SERVING_ENVIRONMENT,
    cpu=SERVING_CPU,
    memory=SERVING_MEMORY,
    timeout=10 * 60,
    volumes={CACHE_PATH: cache_volume},
    max_containers=SERVING_MAX_CONTAINERS,
    scaledown_window=SERVING_SCALEDOWN_WINDOW,
)
@modal.concurrent(max_inputs=SERVING_MAX_INPUTS)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    from app.main import app, query_embedder

    embed = QueryEmbedder().embed.remote
    app.dependency_overrides[query_embedder] = lambda: embed
    return app
