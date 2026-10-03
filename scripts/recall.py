import json
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import modal
import numpy as np
import pyarrow.compute as pc
from sklearn.preprocessing import normalize

from app.config import (
    ANCHOR,
    DATASET_REVISION,
    N_PATCHES,
    N_SPANS,
    NPROBE,
    PROBE,
    SPECTRUM_SURVEYS,
)
from app.search import (
    Query,
    index,
    patches,
    search,
    source,
    spectral,
    spectrum_cells,
)
from app.spectra import spectra
from modal_app import CACHE_PATH, build_image, cache_volume, generate_index
from scripts.benchmark import (
    PATCHES,
    SPANS,
    environment,
    observed_spans,
    queries,
    spec,
)

app = modal.App("alphauniverse-recall")
image = build_image.add_local_python_source("modal_app")

MATCHES = Query.model_fields["matches"].default
REPORT = Path("docs/benchmarks/recall.json")


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


def spectrum_coverage() -> dict[str, np.ndarray]:
    table = source("tokens").to_table(columns=list(SPECTRUM_SURVEYS))
    held = {
        survey: pc.is_valid(table.column(survey)).to_numpy()
        for survey in SPECTRUM_SURVEYS
    }
    return {
        "desi": held["desi"],
        "sdss_only": held["sdss"] & ~held["desi"],
        "none": ~(held["desi"] | held["sdss"]),
    }


def observed_mask() -> np.ndarray:
    cells = spectrum_cells(spectra())
    mask = np.zeros((len(cells), N_SPANS), dtype=bool)
    for galaxy in np.flatnonzero(pc.is_valid(cells).to_numpy()):
        wavelength = np.asarray(cells[int(galaxy)]["wavelength"].values)
        mask[galaxy, observed_spans(wavelength)] = True
    return mask


def match_counts(
    coverage: dict[str, np.ndarray],
    observed: np.ndarray,
    galaxies: np.ndarray,
    patch_maps: np.ndarray,
    span_maps: np.ndarray,
) -> dict[str, Any]:
    spanned = coverage["desi"][galaxies]
    spectral = span_maps[spanned]
    best_is_span = spectral.max(axis=1) > patch_maps[spanned].max(axis=1)
    inside = observed[galaxies[spanned], spectral.argmax(axis=1)]
    return {
        "matches_with": {
            name: int(held[galaxies].sum()) for name, held in coverage.items()
        },
        "best_is_span": int(best_is_span.sum()),
        "best_span_observed": int(inside[best_is_span].sum()),
    }


def with_spans(count: int, holders: np.ndarray) -> list[Query]:
    rng = np.random.default_rng(1)
    cells = spectrum_cells(spectra())
    return [
        Query(
            galaxy=int(galaxy),
            p=tuple(
                int(patch) for patch in rng.choice(N_PATCHES, PATCHES, replace=False)
            ),
            s=tuple(
                int(span)
                for span in rng.choice(
                    observed_spans(np.asarray(cells[int(galaxy)]["wavelength"].values)),
                    SPANS,
                    replace=False,
                )
            ),
        )
        for galaxy in rng.choice(holders, count, replace=False)
    ]


def recall(per_kind: int) -> dict[str, Any]:
    start = time.perf_counter()
    reference = corpus()
    loaded = time.perf_counter() - start
    built = index()
    coverage = spectrum_coverage()
    observed = observed_mask()
    paired = with_spans(per_kind, np.flatnonzero(coverage["desi"]))

    measured = {}
    for kind, batch in (
        ("patches", queries(per_kind, PATCHES, MATCHES)),
        (
            "paired_patches",
            [query.model_copy(update={"spans": ()}) for query in paired],
        ),
        ("spans", [query.model_copy(update={"patches": ()}) for query in paired]),
        ("both", paired),
    ):
        fractions, short, matched = [], [], []
        for query in batch:
            expected = exact_ranking(query, reference)[0][1:]
            found, _, patch_maps, span_maps = (
                part[1:] for part in search(query, index=built)
            )
            fractions.append(len(np.intersect1d(found, expected)) / len(expected))
            short.append(len(found) < len(expected))
            matched.append((found, patch_maps, span_maps))
        measured[kind] = {
            "queries": len(batch),
            "patches": len(batch[0].patches),
            "spans": len(batch[0].spans),
            "mean": round(float(np.mean(fractions)), 4),
            "min": round(float(np.min(fractions)), 4),
            "all_found": round(float(np.mean(np.equal(fractions, 1))), 4),
            "short_of_matches": round(float(np.mean(short)), 4),
            **match_counts(
                coverage,
                observed,
                *(np.concatenate(part) for part in zip(*matched)),
            ),
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
        "galaxies": {name: int(held.sum()) for name, held in coverage.items()},
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
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
