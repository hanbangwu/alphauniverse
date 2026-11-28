import numpy as np

from alphauniverse.text_search.text_search import TextQuery, augmented
from server.dataset.export import identities, written


def text_search_ecsv(galaxies: np.ndarray, scores: np.ndarray, query: TextQuery) -> str:
    table = identities(galaxies)
    table["cosine_similarity"] = scores
    table.meta = {
        "raw_query": query.text,
        "augmented_query": augmented(query.text),
    }
    return written(table)
