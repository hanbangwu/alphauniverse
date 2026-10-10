import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import faiss
import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from app.config import (
    ANCHOR,
    DATASET_REVISION,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    N_TABLE_VALUES,
    REDSHIFT,
    REDSHIFT_TABLE_VALUE,
    SPECTRUM_SURVEYS,
    galaxy_count,
)
from app.dataset import spectrum
from app.main import SPECTRUM_SURVEY
from app.search import (
    BATCH,
    FIRST_HSC_TABLE_VALUE,
    FIRST_LS_TABLE_VALUE,
    N_HSC_TABLE_VALUES,
    N_LS_TABLE_VALUES,
    NPROBE,
    PROBE,
    Query,
    image_tokens,
    index,
    rows,
    search,
    source,
    spectral,
    spectrum_cells,
    with_hsc,
)
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import (
    IMAGE_TOKENS,
    SPECTRUM_TOKENS,
    TABLE_VALUES,
    environment,
    git,
    memory,
    observed_spectrum_tokens,
    queries,
)

app = modal.App("alphauniverse-recall")
image = build_image.add_local_python_source("modal_app")

MATCHES = Query.model_fields["matches"].default
REPORT = Path("docs/benchmarks/search_quality.json")
COLUMNS = [ANCHOR, "hsc", *SPECTRUM_SURVEYS, REDSHIFT]


class Corpus(NamedTuple):
    image_token_rows: np.ndarray
    spectrum_token_rows: np.ndarray
    spectrum_galaxies: np.ndarray
    ls_table_value_rows: np.ndarray
    hsc_table_value_rows: np.ndarray
    hsc_galaxies: np.ndarray
    redshift_rows: np.ndarray
    redshift_galaxies: np.ndarray


def corpus(cells: pa.Table | pa.RecordBatch) -> Corpus:
    spectra = spectrum_cells(cells)
    return Corpus(
        image_tokens(cells.column(ANCHOR)),
        spectral(spectra.drop_null()),
        np.flatnonzero(pc.is_valid(spectra).to_numpy(zero_copy_only=False)),
        rows(cells.column(ANCHOR), N_IMAGE_TOKENS, None),
        rows(cells.column("hsc"), N_IMAGE_TOKENS, None),
        np.flatnonzero(pc.is_valid(cells.column("hsc")).to_numpy(zero_copy_only=False)),
        rows(cells.column(REDSHIFT), 0, None),
        np.flatnonzero(
            pc.is_valid(cells.column(REDSHIFT)).to_numpy(zero_copy_only=False)
        ),
    )


def direction(query: Query, galaxy: int, reference: Corpus) -> np.ndarray:
    owners = np.repeat(reference.spectrum_galaxies, N_SPECTRUM_TOKENS)
    hsc_owners = np.repeat(reference.hsc_galaxies, N_HSC_TABLE_VALUES)
    table_values = np.asarray(query.table_values, dtype=np.int64)
    query_rows = []
    if query.image_tokens:
        query_rows.append(
            reference.image_token_rows[
                galaxy * N_IMAGE_TOKENS + np.asarray(query.image_tokens)
            ]
        )
    if query.spectrum_tokens:
        query_rows.append(
            reference.spectrum_token_rows[owners == galaxy][
                np.asarray(query.spectrum_tokens)
            ]
        )
    ls = table_values[
        (table_values >= FIRST_LS_TABLE_VALUE) & (table_values < FIRST_HSC_TABLE_VALUE)
    ]
    query_rows.append(
        reference.ls_table_value_rows[
            galaxy * N_LS_TABLE_VALUES + ls - FIRST_LS_TABLE_VALUE
        ]
    )
    query_rows.append(
        reference.hsc_table_value_rows[hsc_owners == galaxy][
            table_values[table_values >= FIRST_HSC_TABLE_VALUE] - FIRST_HSC_TABLE_VALUE
        ]
    )
    if REDSHIFT_TABLE_VALUE in query.table_values:
        query_rows.append(
            reference.redshift_rows[reference.redshift_galaxies == galaxy]
        )
    averaged = np.concatenate(query_rows).mean(axis=0, keepdims=True)
    faiss.normalize_L2(averaged)
    return averaged


def exact_maps(
    directions: np.ndarray, reference: Corpus
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(directions)
    galaxies = len(reference.image_token_rows) // N_IMAGE_TOKENS
    image_token_maps = (reference.image_token_rows @ directions.T).reshape(
        galaxies, N_IMAGE_TOKENS, count
    )
    spectral_maps = np.full(
        (galaxies, N_SPECTRUM_TOKENS, count), np.nan, dtype=np.float32
    )
    spectral_maps[reference.spectrum_galaxies] = (
        reference.spectrum_token_rows @ directions.T
    ).reshape(-1, N_SPECTRUM_TOKENS, count)
    table_value_maps = np.full(
        (galaxies, N_TABLE_VALUES, count), np.nan, dtype=np.float32
    )
    table_value_maps[:, FIRST_LS_TABLE_VALUE:FIRST_HSC_TABLE_VALUE] = (
        reference.ls_table_value_rows @ directions.T
    ).reshape(galaxies, N_LS_TABLE_VALUES, count)
    table_value_maps[reference.hsc_galaxies, FIRST_HSC_TABLE_VALUE:] = (
        reference.hsc_table_value_rows @ directions.T
    ).reshape(-1, N_HSC_TABLE_VALUES, count)
    table_value_maps[reference.redshift_galaxies, REDSHIFT_TABLE_VALUE] = (
        reference.redshift_rows @ directions.T
    )
    return image_token_maps, spectral_maps, table_value_maps


def best(
    image_token_maps: np.ndarray,
    spectral_maps: np.ndarray,
    table_value_maps: np.ndarray,
) -> np.ndarray:
    return np.fmax.reduce(
        [
            image_token_maps.max(axis=1),
            spectral_maps.max(axis=1),
            np.fmax.reduce(table_value_maps, axis=1),
        ]
    )


def galaxy_cells(galaxies: np.ndarray) -> pa.Table:
    return source("encoded").to_table(
        columns=COLUMNS, filter=pc.field("galaxy").isin(galaxies)
    )


def exact_rankings(batch: list[Query]) -> list[np.ndarray]:
    galaxies = np.unique([query.galaxy for query in batch])
    taken = corpus(galaxy_cells(galaxies))
    directions = np.concatenate(
        [
            direction(query, int(np.searchsorted(galaxies, query.galaxy)), taken)
            for query in batch
        ]
    )
    del taken
    scores = np.empty((len(batch), source("encoded").count_rows()), dtype=np.float32)
    start = 0
    for cells in source("encoded").to_batches(columns=COLUMNS, batch_size=BATCH):
        scores[:, start : start + cells.num_rows] = best(
            *exact_maps(directions, corpus(cells))
        ).T
        start += cells.num_rows
    rankings = []
    for query, row in zip(batch, scores, strict=True):
        order = np.argsort(-row, kind="stable")
        rankings.append(
            np.concatenate(
                ([query.galaxy], order[order != query.galaxy][: query.matches])
            )
        )
    return rankings


def exact_ranking(
    query: Query,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chosen = exact_rankings([query])[0]
    galaxies = np.sort(chosen)
    taken = corpus(galaxy_cells(galaxies))
    maps = exact_maps(
        direction(query, int(np.searchsorted(galaxies, query.galaxy)), taken), taken
    )
    positions = np.searchsorted(galaxies, chosen)
    return (
        chosen,
        best(*maps)[positions, 0],
        *(found[positions, ..., 0] for found in maps),
    )


def wavelength(galaxy: int) -> np.ndarray:
    tables = (spectrum(galaxy, survey) for survey in SPECTRUM_SURVEYS)
    found = next(table for table in tables if table is not None)
    return found.column("wavelength").to_numpy()


def with_spectrum_tokens(count: int, holders: np.ndarray) -> list[Query]:
    rng = np.random.default_rng(1)
    return [
        Query(
            galaxy=int(galaxy),
            p=tuple(
                int(image_token)
                for image_token in rng.choice(
                    N_IMAGE_TOKENS, IMAGE_TOKENS, replace=False
                )
            ),
            s=tuple(
                int(spectrum_token)
                for spectrum_token in rng.choice(
                    observed_spectrum_tokens(wavelength(int(galaxy))),
                    SPECTRUM_TOKENS,
                    replace=False,
                )
            ),
        )
        for galaxy in rng.choice(holders, count, replace=False)
    ]


def table_values(count: int) -> list[Query]:
    rng = np.random.default_rng(2)
    return [
        Query(
            galaxy=int(rng.integers(galaxy_count())),
            t=tuple(
                (
                    FIRST_LS_TABLE_VALUE
                    + rng.choice(N_LS_TABLE_VALUES, TABLE_VALUES, replace=False)
                ).tolist()
            ),
        )
        for _ in range(count)
    ]


def hsc_table_values(count: int) -> list[Query]:
    rng = np.random.default_rng(3)
    return [
        Query(
            galaxy=int(galaxy),
            t=tuple(
                np.concatenate(
                    (
                        FIRST_LS_TABLE_VALUE
                        + rng.choice(
                            N_LS_TABLE_VALUES, TABLE_VALUES // 2, replace=False
                        ),
                        FIRST_HSC_TABLE_VALUE
                        + rng.choice(
                            N_HSC_TABLE_VALUES, TABLE_VALUES // 2, replace=False
                        ),
                    )
                ).tolist()
            ),
        )
        for galaxy in rng.choice(np.flatnonzero(with_hsc()), count, replace=False)
    ]


@app.function(
    image=image,
    cpu=16,
    memory=(16 * 1024, 64 * 1024),
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_search_quality(per_kind: int) -> dict[str, Any]:
    built = index()
    surveyed = source("tokens").to_table(columns=[SPECTRUM_SURVEY])
    paired = with_spectrum_tokens(
        per_kind,
        np.flatnonzero(pc.is_valid(surveyed.column(SPECTRUM_SURVEY)).to_numpy()),
    )
    kinds = {
        "image_tokens": queries(per_kind, IMAGE_TOKENS, MATCHES),
        "paired_image_tokens": [
            query.model_copy(update={"spectrum_tokens": ()}) for query in paired
        ],
        "spectrum_tokens": [
            query.model_copy(update={"image_tokens": ()}) for query in paired
        ],
        "both": paired,
        "table_values": table_values(per_kind),
        "hsc_table_values": hsc_table_values(per_kind),
    }
    expected = iter(
        exact_rankings([query for batch in kinds.values() for query in batch])
    )

    measured = {}
    for kind, batch in kinds.items():
        fractions, searches = [], []
        for query in batch:
            truth = next(expected)[1:]
            faiss.cvar.indexIVF_stats.reset()
            found = search(query, index=built)[0][1:]
            searches.append(faiss.cvar.indexIVF_stats.nq)
            fractions.append(len(np.intersect1d(found, truth)) / len(truth))
        measured[kind] = {
            "mean": round(float(np.mean(fractions)), 4),
            "min": round(float(np.min(fractions)), 4),
            "all_found": round(float(np.mean(np.equal(fractions, 1))), 4),
            "looked_further": round(float(np.mean(np.greater(searches, 1))), 4),
            "most_searches": max(searches),
        }

    return {
        "environment": environment(),
        "probe": PROBE,
        "nprobe": NPROBE,
        "matches": MATCHES,
        "recall": measured,
        "memory": memory(),
    }


@app.local_entrypoint()
def main(per_kind: int = 100) -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **benchmark_search_quality.remote(per_kind),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
