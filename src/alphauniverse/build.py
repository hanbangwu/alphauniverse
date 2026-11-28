from alphauniverse.encoding.build.encode import generate_embeddings
from alphauniverse.predictions.build.aion_predictions import generate_aion_predictions
from alphauniverse.predictions.build.embeddings import (
    generate_joint_embedding,
    generate_token_embeddings,
)
from alphauniverse.predictions.build.grams import generate_token_grams
from alphauniverse.predictions.build.predictions import generate_predictions
from alphauniverse.text_search.build.alignment import generate_pairs
from alphauniverse.text_search.build.text_regression import (
    generate_text_vectors,
    train_text_regression,
)

STEPS = (
    generate_embeddings,
    generate_predictions,
    generate_token_grams,
    generate_token_embeddings,
    generate_joint_embedding,
    generate_aion_predictions,
    generate_pairs,
    train_text_regression,
    generate_text_vectors,
)


def build() -> None:
    """Build every data product search reads, in order.

    Reads the ``hanbangwu/alphauniverse-cosmos`` dataset and writes into the
    directory named by ``ALPHAUNIVERSE_DATA_DIR``, ``.cache`` by default. Runs
    on the machine it is called on; the AION steps need a GPU.
    """
    for step in STEPS:
        step()
