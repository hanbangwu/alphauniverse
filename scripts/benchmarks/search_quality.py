import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import faiss
import modal
import numpy as np
import pyarrow.compute as pc

from app.config import (
    ANCHOR,
    DATASET_REVISION,
    N_PATCHES,
    N_SCALARS,
    N_SPANS,
    SPECTRUM_SURVEYS,
    galaxy_count,
)
from app.dataset import spectrum
from app.main import SPECTRUM_SURVEY
from app.search import (
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
    observed_spans,
    queries,
)

app = modal.App("alphauniverse-recall")
image = build_image.add_local_python_source("modal_app")

MATCHES = Query.model_fields["matches"].default
REPORT = Path("docs/benchmarks/search_quality.json")


class Corpus(NamedTuple):
    patch_rows: np.ndarray
    span_rows: np.ndarray
    spectrum_galaxies: np.ndarray
    ls_scalar_rows: np.ndarray
    hsc_scalar_rows: np.ndarray
    hsc_galaxies: np.ndarray


def corpus() -> Corpus:
    cells = source("encoded").to_table(columns=[ANCHOR, "hsc", *SPECTRUM_SURVEYS])
    spectra = spectrum_cells(cells)
    return Corpus(
        patches(cells.column(ANCHOR)),
        spectral(spectra.drop_null()),
        np.flatnonzero(pc.is_valid(spectra).to_numpy()),
        rows(cells.column(ANCHOR), N_PATCHES, None),
        rows(cells.column("hsc"), N_PATCHES, None),
        np.flatnonzero(pc.is_valid(cells.column("hsc")).to_numpy()),
    )


def exact_ranking(
    query: Query, reference: Corpus
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    owners = np.repeat(reference.spectrum_galaxies, N_SPANS)
    hsc_owners = np.repeat(reference.hsc_galaxies, N_HSC_SCALARS)
    scalars = np.asarray(query.scalars, dtype=np.int64)
    query_rows = []
    if query.patches:
        query_rows.append(
            reference.patch_rows[query.galaxy * N_PATCHES + np.asarray(query.patches)]
        )
    if query.spans:
        query_rows.append(
            reference.span_rows[owners == query.galaxy][np.asarray(query.spans)]
        )
    query_rows.append(
        reference.ls_scalar_rows[
            query.galaxy * N_LS_SCALARS + scalars[scalars < N_LS_SCALARS]
        ]
    )
    query_rows.append(
        reference.hsc_scalar_rows[hsc_owners == query.galaxy][
            scalars[scalars >= N_LS_SCALARS] - N_LS_SCALARS
        ]
    )
    direction = np.concatenate(query_rows).mean(axis=0, keepdims=True)
    faiss.normalize_L2(direction)
    patch_maps = (reference.patch_rows @ direction.T).reshape(-1, N_PATCHES)
    spectral_maps = np.full((len(patch_maps), N_SPANS), np.nan, dtype=np.float32)
    spectral_maps[reference.spectrum_galaxies] = (
        reference.span_rows @ direction.T
    ).reshape(-1, N_SPANS)
    scalar_maps = np.full((len(patch_maps), N_SCALARS), np.nan, dtype=np.float32)
    scalar_maps[:, :N_LS_SCALARS] = (reference.ls_scalar_rows @ direction.T).reshape(
        -1, N_LS_SCALARS
    )
    scalar_maps[reference.hsc_galaxies, N_LS_SCALARS:] = (
        reference.hsc_scalar_rows @ direction.T
    ).reshape(-1, N_HSC_SCALARS)
    scores = np.fmax.reduce(
        [
            patch_maps.max(axis=1),
            spectral_maps.max(axis=1),
            np.fmax.reduce(scalar_maps, axis=1),
        ]
    )
    order = np.argsort(-scores, kind="stable")
    chosen = np.concatenate(
        ([query.galaxy], order[order != query.galaxy][: query.matches])
    )
    return (
        chosen,
        scores[chosen],
        patch_maps[chosen],
        spectral_maps[chosen],
        scalar_maps[chosen],
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
    reference = corpus()
    built = index()
    surveyed = source("tokens").to_table(columns=[SPECTRUM_SURVEY])
    paired = with_spans(
        per_kind,
        np.flatnonzero(pc.is_valid(surveyed.column(SPECTRUM_SURVEY)).to_numpy()),
    )

    measured = {}
    for kind, batch in (
        ("patches", queries(per_kind, PATCHES, MATCHES)),
        (
            "paired_patches",
            [query.model_copy(update={"spans": ()}) for query in paired],
        ),
        ("spans", [query.model_copy(update={"patches": ()}) for query in paired]),
        ("both", paired),
        ("scalars", scalars(per_kind)),
        ("hsc_scalars", hsc_scalars(per_kind)),
    ):
        fractions, searches = [], []
        for query in batch:
            expected = exact_ranking(query, reference)[0][1:]
            faiss.cvar.indexIVF_stats.reset()
            found = search(query, index=built)[0][1:]
            searches.append(faiss.cvar.indexIVF_stats.nq)
            fractions.append(len(np.intersect1d(found, expected)) / len(expected))
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
