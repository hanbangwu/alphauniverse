import numpy as np

from app.dataset.export import identities, written
from app.text_search.text_search import TextQuery, augmented


def text_search_ecsv(galaxies: np.ndarray, scores: np.ndarray, query: TextQuery) -> str:
    table = identities(galaxies)
    table["cosine_similarity"] = scores
    table.meta = {
        "raw_query": query.text,
        "augmented_query": augmented(query.text),
    }
    return written(table)
