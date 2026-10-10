import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import faiss
import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from app.config import (
    ANCHOR,
    DATASET_REVISION,
    N_PATCHES,
    N_SCALARS,
    N_SPANS,
    REDSHIFT,
    REDSHIFT_SCALAR,
    SPECTRUM_SURVEYS,
    galaxy_count,
)
from app.dataset import spectrum
from app.main import SPECTRUM_SURVEY
from app.search import (
    BATCH,
    N_HSC_SCALARS,
    N_LS_SCALARS,
    NPROBE,
    PROBE,
    Query,
    index,
    patches,
    rows,
    search,
    source,
    spectral,
    spectrum_cells,
    with_hsc,
)
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import (
    PATCHES,
    SCALARS,
    SPANS,
    environment,
    git,
    memory,
    observed_spans,
    queries,
)

app = modal.App("alphauniverse-recall")
image = build_image.add_local_python_source("modal_app")

MATCHES = Query.model_fields["matches"].default
REPORT = Path("docs/benchmarks/search_quality.json")
COLUMNS = [ANCHOR, "hsc", *SPECTRUM_SURVEYS, REDSHIFT]
SCAN = {
    "batch_size": BATCH,
    "batch_readahead": 1,
    "fragment_scan_options": ds.ParquetFragmentScanOptions(pre_buffer=False),
}


class Corpus(NamedTuple):
    patch_rows: np.ndarray
    span_rows: np.ndarray
    spectrum_galaxies: np.ndarray
    ls_scalar_rows: np.ndarray
    hsc_scalar_rows: np.ndarray
    hsc_galaxies: np.ndarray
    redshift_rows: np.ndarray
    redshift_galaxies: np.ndarray


def corpus(cells: pa.Table | pa.RecordBatch) -> Corpus:
    spectra = spectrum_cells(cells)
    return Corpus(
        patches(cells.column(ANCHOR)),
        spectral(spectra.drop_null()),
        np.flatnonzero(pc.is_valid(spectra).to_numpy(zero_copy_only=False)),
        rows(cells.column(ANCHOR), N_PATCHES, None),
        rows(cells.column("hsc"), N_PATCHES, None),
        np.flatnonzero(pc.is_valid(cells.column("hsc")).to_numpy(zero_copy_only=False)),
        rows(cells.column(REDSHIFT), 0, None),
        np.flatnonzero(
            pc.is_valid(cells.column(REDSHIFT)).to_numpy(zero_copy_only=False)
        ),
    )


def direction(query: Query, galaxy: int, reference: Corpus) -> np.ndarray:
    owners = np.repeat(reference.spectrum_galaxies, N_SPANS)
    hsc_owners = np.repeat(reference.hsc_galaxies, N_HSC_SCALARS)
    scalars = np.asarray(query.scalars, dtype=np.int64)
    query_rows = []
    if query.patches:
        query_rows.append(
            reference.patch_rows[galaxy * N_PATCHES + np.asarray(query.patches)]
        )
    if query.spans:
        query_rows.append(
            reference.span_rows[owners == galaxy][np.asarray(query.spans)]
        )
    query_rows.append(
        reference.ls_scalar_rows[
            galaxy * N_LS_SCALARS + scalars[scalars < N_LS_SCALARS]
        ]
    )
    query_rows.append(
        reference.hsc_scalar_rows[hsc_owners == galaxy][
            scalars[(scalars >= N_LS_SCALARS) & (scalars < REDSHIFT_SCALAR)]
            - N_LS_SCALARS
        ]
    )
    if REDSHIFT_SCALAR in query.scalars:
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
    galaxies = len(reference.patch_rows) // N_PATCHES
    patch_maps = (reference.patch_rows @ directions.T).reshape(
        galaxies, N_PATCHES, count
    )
    spectral_maps = np.full((galaxies, N_SPANS, count), np.nan, dtype=np.float32)
    spectral_maps[reference.spectrum_galaxies] = (
        reference.span_rows @ directions.T
    ).reshape(-1, N_SPANS, count)
    scalar_maps = np.full((galaxies, N_SCALARS, count), np.nan, dtype=np.float32)
    scalar_maps[:, :N_LS_SCALARS] = (reference.ls_scalar_rows @ directions.T).reshape(
        galaxies, N_LS_SCALARS, count
    )
    scalar_maps[reference.hsc_galaxies, N_LS_SCALARS:REDSHIFT_SCALAR] = (
        reference.hsc_scalar_rows @ directions.T
    ).reshape(-1, N_HSC_SCALARS, count)
    scalar_maps[reference.redshift_galaxies, REDSHIFT_SCALAR] = (
        reference.redshift_rows @ directions.T
    )
    return patch_maps, spectral_maps, scalar_maps


def best(
    patch_maps: np.ndarray, spectral_maps: np.ndarray, scalar_maps: np.ndarray
) -> np.ndarray:
    return np.fmax.reduce(
        [
            patch_maps.max(axis=1),
            spectral_maps.max(axis=1),
            np.fmax.reduce(scalar_maps, axis=1),
        ]
    )


def galaxy_cells(galaxies: np.ndarray) -> pa.Table:
    return source("encoded").to_table(
        columns=COLUMNS, filter=pc.field("galaxy").isin(galaxies), **SCAN
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
    for cells in source("encoded").to_batches(columns=COLUMNS, **SCAN):
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


def with_spans(count: int, holders: np.ndarray) -> list[Query]:
    rng = np.random.default_rng(1)
    return [
        Query(
            galaxy=int(galaxy),
            p=tuple(
                int(patch) for patch in rng.choice(N_PATCHES, PATCHES, replace=False)
            ),
            s=tuple(
                int(span)
                for span in rng.choice(
                    observed_spans(wavelength(int(galaxy))),
                    SPANS,
                    replace=False,
                )
            ),
        )
        for galaxy in rng.choice(holders, count, replace=False)
    ]


def scalars(count: int) -> list[Query]:
    rng = np.random.default_rng(2)
    return [
        Query(
            galaxy=int(rng.integers(galaxy_count())),
            t=tuple(rng.choice(N_LS_SCALARS, SCALARS, replace=False).tolist()),
        )
        for _ in range(count)
    ]


def hsc_scalars(count: int) -> list[Query]:
    rng = np.random.default_rng(3)
    return [
        Query(
            galaxy=int(galaxy),
            t=tuple(
                np.concatenate(
                    (
                        rng.choice(N_LS_SCALARS, SCALARS // 2, replace=False),
                        N_LS_SCALARS
                        + rng.choice(N_HSC_SCALARS, SCALARS // 2, replace=False),
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
    paired = with_spans(
        per_kind,
        np.flatnonzero(pc.is_valid(surveyed.column(SPECTRUM_SURVEY)).to_numpy()),
    )
    kinds = {
        "patches": queries(per_kind, PATCHES, MATCHES),
        "paired_patches": [query.model_copy(update={"spans": ()}) for query in paired],
        "spans": [query.model_copy(update={"patches": ()}) for query in paired],
        "both": paired,
        "scalars": scalars(per_kind),
        "hsc_scalars": hsc_scalars(per_kind),
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
