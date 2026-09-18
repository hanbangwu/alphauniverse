"""Modal deployment: three build jobs and one serving app.

The build jobs write artifacts into a shared volume mounted at `CACHE_PATH`;
the serving app reads them back. They must run in order — embeddings, then
index, then projections — because each stage consumes the previous one's output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

CACHE_PATH = "/cache"
WANDB_MODE = "offline"

app = modal.App("alphauniverse")

cache_volume = modal.Volume.from_name("alphauniverse-cache", create_if_missing=True)

environment = {
    "ALPHAUNIVERSE_CACHE": CACHE_PATH,
    "HF_HOME": CACHE_PATH,
    "WANDB_DIR": CACHE_PATH,
    "WANDB_MODE": WANDB_MODE,
}

# --no-dev: uv's default group holds the test runner, which no image needs.
serving_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-dev")
    .env(environment)
    .add_local_python_source("app")
)

build_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["build"], extra_options="--no-dev")
    .env(environment)
    .add_local_python_source("app")
)


@app.function(
    image=build_image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_embeddings() -> None:
    """Stage 1: encode every galaxy into the three parquet stores."""
    from app.encode import generate_embeddings

    cache_volume.reload()
    generate_embeddings()
    cache_volume.commit()


@app.function(
    image=build_image,
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_index() -> None:
    """Stage 2: build the patch search index from `encoded`."""
    from app.search import generate_index

    cache_volume.reload()
    generate_index()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
    secrets=(
        [modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])]
        if WANDB_MODE == "online"
        else []
    ),
)
def generate_projections() -> None:
    """Stage 3: fit the projector and write both point sets."""
    from app.parametric_umap import generate_projections

    cache_volume.reload()
    generate_projections()
    cache_volume.commit()


@app.function(
    image=serving_image,
    cpu=8,
    memory=(8 * 1024, 64 * 1024),
    timeout=10 * 60,
    volumes={CACHE_PATH: cache_volume},
    max_containers=1,
    scaledown_window=5 * 60,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def fastapi_app() -> FastAPI:
    """Serve the read-only API."""
    from app.main import app

    return app
