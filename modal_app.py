from __future__ import annotations

from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

CACHE_PATH = "/cache"
WANDB_MODE = "offline"
SERVING_CPU = 8
SERVING_MEMORY = (8 * 1024, 64 * 1024)
SERVING_MAX_INPUTS = 16
SERVING_MAX_CONTAINERS = 1
SERVING_SCALEDOWN_WINDOW = 5 * 60

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
    from app.config import build_dir
    from app.encode import generate_embeddings

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_embeddings()


@app.function(
    image=build_image,
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_index() -> None:
    from app.config import build_dir
    from app.search import generate_index

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_index()


@app.function(
    image=serving_image,
    cpu=1,
    memory=(8 * 1024, 32 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_images() -> None:
    from app.config import build_dir
    from app.images import generate_images

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_images()


@app.function(
    image=serving_image,
    cpu=1,
    memory=(8 * 1024, 32 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_spectra() -> None:
    from app.config import build_dir
    from app.spectra import generate_spectra

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_spectra()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_projections() -> None:
    from app.config import build_dir
    from app.parametric_umap import generate_projections

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_projections()


@app.function(
    image=serving_image,
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
    from app.main import app

    return app
