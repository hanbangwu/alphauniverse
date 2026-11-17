import numpy as np

from app.dataset.export import identities, written


def text_search_csv(galaxies: np.ndarray, scores: np.ndarray) -> str:
    table = identities(galaxies)
    table["cosine_similarity"] = scores
    return written(table, "ascii.csv")
