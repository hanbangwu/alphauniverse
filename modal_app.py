from __future__ import annotations

from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
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
    gpu="T4",
    cpu=(2, 4),
    memory=(8 * 1024, 32 * 1024),
    timeout=2 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_embeddings() -> None:
    from app.config import build_dir
    from app.encode import generate_embeddings

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_embeddings()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="T4",
    cpu=(8, 16),
    memory=(4 * 1024, 16 * 1024),
    timeout=4 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_predictions() -> None:
    from app.config import build_dir
    from app.predictions import generate_predictions

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_predictions()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="H100!",
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=4 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_token_grams() -> None:
    from app.grams import generate_token_grams

    generate_token_grams()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="H100!",
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=5 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_token_embeddings() -> None:
    from app.embeddings import generate_token_embeddings

    generate_token_embeddings()
    cache_volume.commit()


@app.function(
    image=build_image,
    cpu=2,
    memory=(2 * 1024, 8 * 1024),
    timeout=1 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_joint_embedding() -> None:
    from app.embeddings import generate_joint_embedding

    generate_joint_embedding()
    cache_volume.commit()


@app.function(
    image=build_image,
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=2 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_properties() -> None:
    from app.properties import generate_properties

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
    from app.maps import generate_maps

    generate_maps()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=20 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_templates() -> None:
    from app.templates import generate_templates

    generate_templates()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="H100!",
    cpu=4,
    memory=(16 * 1024, 64 * 1024),
    timeout=80 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_observed_grams() -> None:
    from app.observed_grams import generate_observed_grams

    generate_observed_grams()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="H100!",
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=10 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_template_embeddings() -> None:
    from app.observed_grams import generate_template_embeddings

    generate_template_embeddings()
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
    from app.config import build_dir
    from app.parametric_umap import generate_projections

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_projections()
    cache_volume.commit()


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
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=4,
    memory=(32 * 1024, 64 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def generate_alignment() -> None:
    from app.alignment import generate_alignment

    generate_alignment()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L4",
    cpu=4,
    memory=(32 * 1024, 64 * 1024),
    timeout=1 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_aion_gemma_space() -> None:
    from app.alignment import generate_aion_gemma_space

    generate_aion_gemma_space()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L40S",
    cpu=8,
    memory=(16 * 1024, 64 * 1024),
    timeout=24 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_text_views(views: int) -> None:
    from app.text_views import generate_text_views

    generate_text_views(views)
    cache_volume.commit()


serving_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-dev")
    .env(environment | {"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("app")
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
    from app.main import app

    return app
