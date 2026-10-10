from __future__ import annotations

from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

CACHE_PATH = "/cache"
SERVING_CPU = 8
SERVING_MEMORY = (8 * 1024, 32 * 1024)
SERVING_MAX_INPUTS = 16
SERVING_MAX_CONTAINERS = 1
SERVING_SCALEDOWN_WINDOW = 5 * 60

app = modal.App("alphauniverse")

cache_volume = modal.Volume.from_name("alphauniverse-cache", create_if_missing=True)

environment = {
    "ALPHAUNIVERSE_CACHE": CACHE_PATH,
    "HF_HOME": CACHE_PATH,
    "WANDB_DIR": CACHE_PATH,
}


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
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=24 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_predictions() -> None:
    from app.config import build_dir
    from app.predictions import generate_predictions

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_predictions()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def generate_projections() -> None:
    from app.config import build_dir
    from app.parametric_umap import generate_projections

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_projections()


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
    image=build_image,
    gpu="L4",
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=6 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_pairs() -> None:
    from app.config import build_dir
    from app.text_search import generate_pairs

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_pairs()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def generate_alignment() -> None:
    from app.alignment import generate_alignment

    generate_alignment()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_aion_gemma_space() -> None:
    from app.alignment import generate_aion_gemma_space

    generate_aion_gemma_space()


serving_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-dev")
    .env(environment | {"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("app")
)


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
