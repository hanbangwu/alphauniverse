from __future__ import annotations

from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    import numpy as np
    from fastapi import FastAPI

CACHE_PATH = "/cache"
SERVING_CPU = 8
SERVING_MEMORY = (4 * 1024, 32 * 1024)
SERVING_MAX_INPUTS = 16
SERVING_MAX_CONTAINERS = 1
SERVING_SCALEDOWN_WINDOW = 5 * 60
SERVING_ENVIRONMENT = {"OPENBLAS_NUM_THREADS": "1"}

app = modal.App("alphauniverse")

cache_volume = modal.Volume.from_name("alphauniverse-cache", create_if_missing=True)

environment = {
    "ALPHAUNIVERSE_DATA_DIR": CACHE_PATH,
    "HF_HOME": CACHE_PATH,
    "WANDB_DIR": CACHE_PATH,
}


build_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["build"], extra_options="--no-dev")
    .env(environment)
    .add_local_python_source("alphauniverse", "server")
)


@app.function(
    image=build_image,
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_properties() -> None:
    from server.projection.build.properties import generate_properties

    generate_properties()
    cache_volume.commit()


@app.function(
    image=build_image,
    cpu=8,
    memory=(8 * 1024, 32 * 1024),
    timeout=5 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_maps() -> None:
    from server.projection.build.maps import generate_maps

    generate_maps()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=4,
    memory=(32 * 1024, 64 * 1024),
    timeout=6 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def generate_projections() -> None:
    from server.projection.build.parametric_umap import generate_projections

    generate_projections()
    cache_volume.commit()


@app.cls(
    image=build_image,
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
        from alphauniverse.text_search.text_search import embed_queries

        embed_queries(["a galaxy"])

    @modal.method()
    def embed(self, texts: list[str]) -> np.ndarray:
        from alphauniverse.text_search.text_search import embed_queries

        return embed_queries(texts)


serving_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["server"], extra_options="--no-dev")
    .env(environment | {"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("alphauniverse", "server")
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
    secrets=[modal.Secret.from_name("access-password")],
)
@modal.concurrent(max_inputs=SERVING_MAX_INPUTS)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    from server.main import app
    from server.text_search.routes import query_embedder

    embed = QueryEmbedder().embed.remote
    app.dependency_overrides[query_embedder] = lambda: embed
    return app
