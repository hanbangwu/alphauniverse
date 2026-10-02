import json
import resource
import subprocess
import time
from datetime import UTC, datetime
from typing import Any, NamedTuple

import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from sklearn.preprocessing import normalize

from app.config import (
    ANCHOR,
    DATASET_REVISION,
    N_PATCHES,
    N_SPANS,
    NPROBE,
    PROBE,
    SPECTRUM_ORIGIN,
    SPECTRUM_SURVEYS,
    SPECTRUM_TOKEN_WIDTH,
)
from app.search import (
    Query,
    index,
    patches,
    search,
    source,
    spectral,
    spectrum_cells,
    with_spectrum,
)
from app.spectra import spectra
from modal_app import CACHE_PATH, build_image, cache_volume, generate_index
from scripts.benchmark import PATCHES, environment, queries, spec

app = modal.App("alphauniverse-recall")
image = build_image.add_local_python_source("modal_app")

MATCHES = Query.model_fields["matches"].default
SPANS = 4


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


def observed_spans(cell: pa.StructScalar) -> np.ndarray:
    wavelength = np.asarray(cell["wavelength"].values)
    first, last = np.floor(
        (np.array([wavelength.min(), wavelength.max()]) - SPECTRUM_ORIGIN)
        / SPECTRUM_TOKEN_WIDTH
    ).astype(int)
    return np.arange(max(first, 0), min(last, N_SPANS - 1) + 1)


def with_spans(count: int, patch_count: int, seed: int) -> list[Query]:
    rng = np.random.default_rng(seed)
    cells = spectrum_cells(spectra())
    return [
        Query(
            galaxy=int(galaxy),
            p=tuple(
                int(patch)
                for patch in rng.choice(N_PATCHES, patch_count, replace=False)
            ),
            s=tuple(
                int(span)
                for span in rng.choice(
                    observed_spans(cells[int(galaxy)]), SPANS, replace=False
                )
            ),
        )
        for galaxy in rng.choice(np.flatnonzero(with_spectrum()), count)
    ]


def recall(per_kind: int) -> dict[str, Any]:
    start = time.perf_counter()
    reference = corpus()
    loaded = time.perf_counter() - start
    built = index()

    measured = {}
    for kind, batch in (
        ("patches", queries(per_kind, PATCHES, MATCHES)),
        ("spans", with_spans(per_kind, 0, seed=1)),
        ("both", with_spans(per_kind, PATCHES, seed=2)),
    ):
        fractions, short = [], []
        for query in batch:
            expected = exact_ranking(query, reference)[0][1:]
            found = search(query, index=built)[0][1:]
            fractions.append(len(np.intersect1d(found, expected)) / len(expected))
            short.append(len(found) < len(expected))
        measured[kind] = {
            "queries": len(batch),
            "patches": len(batch[0].patches),
            "spans": len(batch[0].spans),
            "mean": round(float(np.mean(fractions)), 4),
            "min": round(float(np.min(fractions)), 4),
            "all_found": round(float(np.mean(np.equal(fractions, 1))), 4),
            "short_of_matches": round(float(np.mean(short)), 4),
        }

    return {
        "environment": environment(),
        "probe": PROBE,
        "nprobe": NPROBE,
        "matches": MATCHES,
        "corpus_load_s": round(loaded, 1),
        "total_s": round(time.perf_counter() - start, 1),
        "peak_rss_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9, 2
        ),
        "recall": measured,
    }


@app.function(
    image=image,
    cpu=generate_index.spec.cpu,
    memory=generate_index.spec.memory,
    timeout=3 * 60 * 60,
    volumes={CACHE_PATH: cache_volume.with_mount_options(read_only=True)},
)
def measure(per_kind: int) -> dict[str, Any]:
    return recall(per_kind)


@app.local_entrypoint()
def main(per_kind: int = 100) -> None:
    report = {
        "commit": subprocess.check_output(
            ["git", "describe", "--always", "--dirty"], text=True
        ).strip(),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "job": spec(measure),
        **measure.remote(per_kind),
    }
    print(json.dumps(report, indent=2))
