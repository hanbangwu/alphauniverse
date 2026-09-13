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

serving_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .env(environment)
    .add_local_python_source("app")
)

build_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["build"])
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
    from app.main import app

    return app
