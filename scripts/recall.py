from typing import NamedTuple

import numpy as np
import pyarrow.compute as pc
from sklearn.preprocessing import normalize

from app.config import ANCHOR, N_PATCHES, N_SPANS, SPECTRUM_SURVEYS
from app.search import Query, patches, source, spectral, spectrum_cells


class Corpus(NamedTuple):
    patch_rows: np.ndarray
    span_rows: np.ndarray
    spectrum_galaxies: np.ndarray


def corpus() -> Corpus:
    cells = source("encoded").to_table(columns=[ANCHOR, *SPECTRUM_SURVEYS])
    spectra = spectrum_cells(cells)
    return Corpus(
        patches(cells.column(ANCHOR).combine_chunks()),
        spectral(spectra.drop_null()),
        np.flatnonzero(pc.is_valid(spectra).to_numpy(zero_copy_only=False)),
    )


def exact_ranking(
    query: Query, reference: Corpus
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    owners = np.repeat(reference.spectrum_galaxies, N_SPANS)
    query_rows = []
    if query.patches:
        query_rows.append(
            reference.patch_rows[query.galaxy * N_PATCHES + np.asarray(query.patches)]
        )
    if query.spans:
        query_rows.append(
            reference.span_rows[owners == query.galaxy][np.asarray(query.spans)]
        )
    direction = normalize(np.concatenate(query_rows).mean(axis=0, keepdims=True))
    patch_maps = (reference.patch_rows @ direction.T).reshape(-1, N_PATCHES)
    spectral_maps = np.full((len(patch_maps), N_SPANS), np.nan, dtype=np.float32)
    spectral_maps[reference.spectrum_galaxies] = (
        reference.span_rows @ direction.T
    ).reshape(-1, N_SPANS)
    scores = np.fmax(patch_maps.max(axis=1), spectral_maps.max(axis=1))
    order = np.argsort(-scores, kind="stable")
    chosen = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    return chosen, scores[chosen], patch_maps[chosen], spectral_maps[chosen]
