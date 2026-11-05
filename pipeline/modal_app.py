import modal

from core.modal_resources import CACHE_PATH, build_dependencies, cache_volume

app = modal.App("alphauniverse-build")

build_image = build_dependencies.add_local_python_source("core", "pipeline")


@app.function(
    image=build_image,
    gpu="T4",
    cpu=(2, 4),
    memory=(8 * 1024, 32 * 1024),
    timeout=2 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_embeddings() -> None:
    from core.artifacts import build_dir
    from pipeline.encode import generate_embeddings

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
    from core.artifacts import build_dir
    from pipeline.predictions import generate_predictions

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
    from pipeline.grams import generate_token_grams

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
    from pipeline.embeddings import generate_token_embeddings

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
    from pipeline.embeddings import generate_joint_embedding

    generate_joint_embedding()
    cache_volume.commit()


@app.function(
    image=build_image,
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_properties() -> None:
    from pipeline.properties import generate_properties

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
    from pipeline.maps import generate_maps

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
def generate_aion_predictions() -> None:
    from pipeline.aion_predictions import generate_aion_predictions

    generate_aion_predictions()
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
    from pipeline.observed_grams import generate_observed_grams

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
def generate_aion_prediction_embeddings() -> None:
    from pipeline.observed_grams import generate_aion_prediction_embeddings

    generate_aion_prediction_embeddings()
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
    from core.artifacts import build_dir
    from pipeline.parametric_umap import generate_projections

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_projections()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L40S",
    cpu=4,
    memory=(8 * 1024, 32 * 1024),
    timeout=6 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_pairs() -> None:
    from core.artifacts import build_dir
    from pipeline.text_pairs import generate_pairs

    build_dir().mkdir(parents=True, exist_ok=True)
    generate_pairs()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L40S",
    cpu=16,
    memory=(16 * 1024, 64 * 1024),
    timeout=24 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_text_views(parts: int) -> None:
    from pipeline.text_views import generate_text_views

    generate_text_views(parts)
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L40S",
    cpu=16,
    memory=(16 * 1024, 64 * 1024),
    timeout=24 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_text_regression() -> None:
    from pipeline.text_regression import train_text_regression

    train_text_regression()
    cache_volume.commit()


@app.function(
    image=build_image,
    gpu="L40S",
    cpu=4,
    memory=(16 * 1024, 32 * 1024),
    timeout=1 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def generate_text_vectors() -> None:
    from pipeline.text_regression import generate_text_vectors

    generate_text_vectors()
    cache_volume.commit()
