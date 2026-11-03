import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import modal
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from app.alignment import held_out, pairs
from app.config import (
    DATASET_REVISION,
    MODES,
    SEED,
    TEXT_SURVEYS,
    VIEW_COMBINATIONS,
    artifact,
    galaxy_count,
)
from app.text_search import (
    FRAMES,
    TextQuery,
    embed_queries,
    generic_queries,
    query_directions,
    text_vectors,
)
from app.text_views import training
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-text-search")
image = build_image.add_local_python_source("modal_app")


class Pair(NamedTuple):
    label: str
    mode: str | None
    high: tuple[str, ...]
    low: tuple[str, ...]


PAIRS = {
    "colour": Pair(
        "g_r",
        "images",
        (
            "a red galaxy",
            "a galaxy with red colours",
            "a reddish galaxy",
            "an orange-red galaxy",
            "a galaxy that looks red",
            "a galaxy with a red hue",
        ),
        (
            "a blue galaxy",
            "a galaxy with blue colours",
            "a bluish galaxy",
            "a blue-white galaxy",
            "a galaxy that looks blue",
            "a galaxy with a blue hue",
        ),
    ),
    "size": Pair(
        "radius_measured",
        "images",
        (
            "a large galaxy",
            "a big galaxy",
            "an extended galaxy",
            "a galaxy with a large apparent size",
            "a galaxy that fills the frame",
            "a sprawling galaxy",
        ),
        (
            "a small galaxy",
            "a tiny galaxy",
            "a compact galaxy",
            "a galaxy with a small apparent size",
            "a galaxy that appears as a small dot",
            "a point-like galaxy",
        ),
    ),
    "brightness": Pair(
        "r_magnitude_measured",
        "images",
        (
            "a faint galaxy",
            "a very faint galaxy",
            "a dim galaxy",
            "a galaxy that appears faint",
            "a barely visible galaxy",
            "a faint, dim smudge",
        ),
        (
            "a bright galaxy",
            "a very bright galaxy",
            "a brilliant galaxy",
            "a galaxy that appears bright",
            "a galaxy that shines brightly",
            "a bright, prominent galaxy",
        ),
    ),
    "shape": Pair(
        "axis_ratio_measured",
        "images",
        (
            "a round galaxy",
            "a face-on galaxy",
            "a circular galaxy",
            "a galaxy seen face-on",
            "a galaxy with a round shape",
            "a galaxy that looks circular",
        ),
        (
            "an edge-on galaxy",
            "an edge-on disk galaxy",
            "a thin, elongated galaxy",
            "a galaxy seen edge-on",
            "a cigar-shaped galaxy",
            "a needle-like galaxy",
        ),
    ),
    "emission": Pair(
        "halpha_ew_measured",
        "spectra",
        (
            "a galaxy with strong emission lines",
            "an emission-line galaxy",
            "a galaxy with strong H-alpha emission",
            "a galaxy with bright, narrow emission lines",
            "a galaxy with prominent emission lines",
            "a galaxy with strong [OII] and H-alpha lines",
        ),
        (
            "an absorption-line galaxy",
            "a galaxy with strong absorption lines",
            "a galaxy with weak emission lines",
            "a galaxy dominated by absorption lines",
            "a galaxy with deep absorption features",
            "a galaxy with very weak emission lines",
        ),
    ),
    "break": Pair(
        "dn4000_measured",
        "spectra",
        (
            "a galaxy with a strong 4000 angstrom break",
            "a galaxy with an old stellar population",
            "a galaxy made of old stars",
            "a galaxy with a large break near 4000 angstroms",
            "an old, evolved galaxy",
            "a galaxy with an aged stellar population",
        ),
        (
            "a galaxy with a weak 4000 angstrom break",
            "a galaxy with a young stellar population",
            "a galaxy made of young stars",
            "a galaxy with a small break near 4000 angstroms",
            "a young galaxy",
            "a galaxy with a youthful stellar population",
        ),
    ),
    "redshift": Pair(
        "redshift_measured",
        None,
        (
            "a distant galaxy",
            "a galaxy at high redshift",
            "a faraway galaxy",
            "a high-redshift galaxy",
            "a very distant galaxy",
            "a remote galaxy",
        ),
        (
            "a nearby galaxy",
            "a galaxy at low redshift",
            "a local galaxy",
            "a low-redshift galaxy",
            "a very nearby galaxy",
            "a galaxy close to us",
        ),
    ),
    "star formation": Pair(
        "ssfr_measured",
        None,
        (
            "a star-forming galaxy",
            "a galaxy actively forming stars",
            "a galaxy with a high star formation rate",
            "a galaxy with ongoing star formation",
            "a galaxy forming many new stars",
            "a galaxy with vigorous star formation",
        ),
        (
            "a quiescent galaxy",
            "a passive galaxy",
            "a galaxy that has stopped forming stars",
            "a red and dead galaxy",
            "a quenched galaxy",
            "a galaxy with a low star formation rate",
        ),
    ),
    "mass": Pair(
        "mass_measured",
        None,
        (
            "a massive galaxy",
            "a very massive galaxy",
            "a galaxy with a high stellar mass",
            "a giant galaxy",
            "a galaxy with a large stellar mass",
            "a giant, massive galaxy",
        ),
        (
            "a dwarf galaxy",
            "a low-mass galaxy",
            "a galaxy with a low stellar mass",
            "a small, low-mass galaxy",
            "a galaxy with a small stellar mass",
            "a lightweight galaxy",
        ),
    ),
}
MORPHOLOGY = {
    "spiral": (
        "a spiral galaxy",
        "a galaxy with spiral arms",
        "a face-on spiral galaxy",
        "a grand-design spiral galaxy",
        "a disk galaxy with spiral structure",
        "a barred spiral galaxy",
    ),
    "elliptical": (
        "an elliptical galaxy",
        "a smooth elliptical galaxy",
        "a featureless elliptical galaxy",
        "a round elliptical galaxy",
        "an early-type galaxy",
        "a smooth, round galaxy with no features",
    ),
    "edge-on": (
        "an edge-on disk galaxy",
        "an edge-on spiral galaxy",
        "a disk galaxy seen edge-on",
        "a thin edge-on disk",
        "an edge-on galaxy with a dust lane",
        "a spiral galaxy seen from the side",
    ),
    "merger": (
        "a merging galaxy with tidal tails",
        "two interacting galaxies",
        "a galaxy merger",
        "a disturbed galaxy with tidal features",
        "a pair of colliding galaxies",
        "an interacting galaxy",
    ),
    "irregular": (
        "an irregular galaxy",
        "a clumpy irregular galaxy",
        "a galaxy with an irregular shape",
        "a messy, asymmetric galaxy",
        "a galaxy with no regular structure",
        "a patchy irregular galaxy",
    ),
}
HALVES = ("development", "sealed")
SIDES = {"high": "low", "low": "high"}
MODE_SETS = {name: VIEW_COMBINATIONS[name] for name in FRAMES}
RANDOM_DIRECTIONS = 200
EDITS = 50
TOP = TextQuery.model_fields["matches"].default
PAGE = 8
GALAXIES = 1024
REPORT = Path("docs/benchmarks/text_search_quality.json")


type Scorer = Callable[[np.ndarray, Any], dict[str, np.ndarray]]


def centred(vectors: np.ndarray) -> np.ndarray:
    return normalize(vectors - vectors.mean(axis=0))


def view_scorer(
    found: dict[str, tuple[np.ndarray, np.ndarray]], sets: dict[str, tuple[str, ...]]
) -> Scorer:
    def score(directions: np.ndarray, galaxies: Any) -> dict[str, np.ndarray]:
        scores = {}
        for name, modalities in sets.items():
            total = np.zeros((len(directions), galaxy_count()))
            count = np.zeros(galaxy_count())
            for modality in modalities:
                galaxy, vectors = found[modality]
                total[:, galaxy] += directions @ vectors.T
                count[galaxy] += 1
            mean = np.where(count > 0, total / np.maximum(count, 1), -1.0)
            scores[name] = mean[:, galaxies]
        return scores

    return score


def set_vectors(
    modes: np.ndarray, sets: dict[str, tuple[str, ...]]
) -> dict[str, np.ndarray]:
    columns = {
        name: [MODES.index(mode) for mode in found] for name, found in sets.items()
    }
    vectors = {
        name: np.empty((len(modes), modes.shape[2]), dtype=np.float32) for name in sets
    }
    for start in range(0, len(modes), GALAXIES):
        rows = np.asarray(modes[start : start + GALAXIES])
        for name, kept in columns.items():
            vectors[name][start : start + GALAXIES] = rows[:, kept].sum(axis=1)
    return vectors


def search_scorer(vectors: dict[str, np.ndarray]) -> Scorer:
    def score(directions: np.ndarray, galaxies: Any) -> dict[str, np.ndarray]:
        return {name: directions @ found[galaxies].T for name, found in vectors.items()}

    return score


def families(views: dict) -> list[tuple[str, Scorer, str]]:
    stored = text_vectors()
    parts = {name: modes for name, modes in MODE_SETS.items() if modes != MODES}
    vectors = set_vectors(stored.modes, parts) | {
        name: stored.vectors.numpy()
        for name, modes in MODE_SETS.items()
        if modes == MODES
    }
    found = [("search", search_scorer(vectors), "plain")]
    for name in MODE_SETS:
        found.append(("search framed", search_scorer({name: vectors[name]}), name))
    modalities = {
        name: tuple(
            modality
            for modality, survey in TEXT_SURVEYS.items()
            if f"{survey}_{modality}" in modes
        )
        for name, modes in MODE_SETS.items()
    }
    observed = {name: chosen for name, chosen in modalities.items() if chosen}
    found.append(("views", view_scorer(views, observed), "plain"))
    return found


def sides(values: np.ndarray) -> dict[str, np.ndarray]:
    low, high = np.nanquantile(values, [1 / 3, 2 / 3])
    return {"high": values > high, "low": values < low, "known": np.isfinite(values)}


def page_sides(measured: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    low, high = np.nanquantile(measured, [1 / 3, 2 / 3])
    values = np.where(np.isfinite(measured), measured, predicted)
    return {"high": values > high, "low": values < low}


def edited(first: np.ndarray, second: np.ndarray, steps: np.ndarray) -> np.ndarray:
    middle = (first + second) / 2
    scaled = steps * np.linalg.norm(first - second) / 2
    return np.concatenate([normalize(middle + scaled), normalize(middle - scaled)])


def top(score: np.ndarray, count: int = TOP) -> np.ndarray:
    return np.argsort(-score)[:count]


def image_percentile(
    score: np.ndarray, reference: np.ndarray, imaged: np.ndarray, count: int
) -> float:
    percentiles = rankdata(reference[imaged]) / len(imaged)
    return round(float(percentiles[top(score[imaged], count)].mean()), 4)


def morphology_rows(
    by_text: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    imaged: np.ndarray,
    chosen: dict[str, tuple[str, ...]],
    described: dict[str, np.ndarray],
    trained: np.ndarray,
) -> dict[str, Any]:
    pages = {
        text: top(by_text[text], PAGE) for found in chosen.values() for text in found
    }
    overlaps = {
        f"{first}|{second}": round(
            float(
                np.mean(
                    [
                        len(np.intersect1d(pages[one], pages[other])) / PAGE
                        for one, other in zip(chosen[first], chosen[second])
                    ]
                )
            ),
            4,
        )
        for index, first in enumerate(chosen)
        for second in list(chosen)[index + 1 :]
    }
    return {
        f"mean_overlap@{PAGE}": round(float(np.mean(list(overlaps.values()))), 4),
        f"overlap@{PAGE}": overlaps,
        "phrasings": {
            text: {
                f"top{PAGE}": page.tolist(),
                f"image_percentile@{PAGE}": image_percentile(
                    by_text[text], reference[text], imaged, PAGE
                ),
                f"trained@{PAGE}": int(np.isin(page, trained).sum()),
            }
            | {
                f"median_{name}@{PAGE}": round(float(np.nanmedian(values[page])), 4)
                for name, values in described.items()
            }
            for text, page in pages.items()
        },
    }


def precision(
    labels: dict[str, np.ndarray], held: np.ndarray, side: str, score: np.ndarray
) -> float:
    inside = labels["known"][held]
    return float(average_precision_score(labels[side][held][inside], score[inside]))


def separation(
    scores: dict[str, np.ndarray],
    both: tuple[str, str],
    labels: dict[str, np.ndarray],
    held: np.ndarray,
) -> float:
    found = [
        precision(labels, held, side, scores[own][held])
        - precision(labels, held, side, scores[other][held])
        for side, own, other in zip(SIDES, both, both[::-1])
    ]
    return round(sum(found) / len(found), 4)


def pair_rows(
    scores: dict[str, np.ndarray],
    edits: dict[tuple[str, str], np.ndarray],
    random_scores: np.ndarray,
    phrasings: tuple[tuple[str, str], ...],
    labels: dict[str, np.ndarray],
    estimated: dict[str, np.ndarray],
    held: np.ndarray,
    reference: dict[str, np.ndarray],
    imaged: np.ndarray,
) -> dict[str, dict[str, dict[str, Any]]]:
    ranked = partial(precision, labels, held)
    known = labels["known"][held]
    rates = {side: labels[side][held][known].mean() for side in SIDES}
    nulls = {
        side: np.quantile([ranked(side, row) for row in random_scores], 0.95)
        for side in SIDES
    }
    rows: dict[str, dict[str, dict[str, Any]]] = {side: {} for side in SIDES}
    for both in phrasings:
        found = {side: scores[text] for side, text in zip(SIDES, both)}
        plus, minus = np.split(edits[both], 2)
        for (side, opposite), text in zip(SIDES.items(), both):
            own = ranked(side, found[side][held])
            specificity = own - ranked(side, found[opposite][held])
            edit_null = np.quantile(
                [
                    abs(ranked(side, first) - ranked(side, second))
                    for first, second in zip(plus, minus)
                ],
                0.95,
            )
            page = top(np.where(labels["known"], found[side], -np.inf))
            shown = top(found[side])
            row = {
                "average_precision": round(own, 4),
                "base_rate": round(float(rates[side]), 4),
                "specificity": round(specificity, 4),
                "edit_null": round(float(edit_null), 4),
                "random_null": round(float(nulls[side]), 4),
                "passes": bool(specificity > edit_null and own > rates[side]),
                f"precision@{TOP}": round(float(labels[side][page].mean()), 4),
                f"opposite_precision@{TOP}": round(
                    float(labels[opposite][page].mean()), 4
                ),
                f"page_precision@{TOP}": round(float(estimated[side][shown].mean()), 4),
                f"page_opposite_precision@{TOP}": round(
                    float(estimated[opposite][shown].mean()), 4
                ),
                f"page_measured@{TOP}": round(float(labels["known"][shown].mean()), 4),
                f"image_percentile@{TOP}": image_percentile(
                    found[side], reference[text], imaged, TOP
                ),
            }
            rows[side][text] = row
    return rows


@app.function(
    image=image,
    gpu="L40S",
    cpu=16,
    memory=(16 * 1024, 64 * 1024),
    timeout=2 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_text_search_quality(half: str) -> dict[str, Any]:
    held = np.flatnonzero(held_out(np.arange(galaxy_count())).cpu().numpy())
    start = HALVES.index(half)
    chosen = {
        name: tuple(zip(pair.high[start::2], pair.low[start::2]))
        for name, pair in PAIRS.items()
    }
    phrased = [both for found in chosen.values() for both in found]
    shapes = {name: found[start::2] for name, found in MORPHOLOGY.items()}
    texts = [text for both in phrased for text in both] + [
        text for found in shapes.values() for text in found
    ]

    generic = embed_queries(generic_queries())
    queries = {
        "plain": query_directions(embed_queries(texts), text_vectors().query_mean)
    }
    for (name, frame), block in zip(FRAMES.items(), np.split(generic, len(FRAMES))):
        framed = embed_queries([frame.format(text) for text in texts])
        queries[name] = query_directions(framed, torch.from_numpy(block.mean(axis=0)))
    queries = {name: rows.numpy() for name, rows in queries.items()}
    rng = np.random.default_rng(SEED)
    random_directions = normalize(
        rng.standard_normal((RANDOM_DIRECTIONS, generic.shape[1]))
    )
    steps = normalize(rng.standard_normal((EDITS, generic.shape[1])))

    views = {}
    for modality in TEXT_SURVEYS:
        galaxy, _, gemma = pairs(modality)
        views[modality] = galaxy, centred(gemma.cpu().numpy())
    imaged = np.intersect1d(held, views["image"][0])
    reference = dict(
        zip(
            texts,
            view_scorer(views, {"images": ("image",)})(queries["plain"], slice(None))[
                "images"
            ],
        )
    )

    properties = pq.read_table(artifact("properties"))
    measured = {
        name: properties.column(pair.label).to_numpy(zero_copy_only=False)
        for name, pair in PAIRS.items()
    }
    predicted = {
        name: properties.column(pair.label.replace("_measured", "_predicted")).to_numpy(
            zero_copy_only=False
        )
        if pair.label.endswith("_measured")
        else measured[name]
        for name, pair in PAIRS.items()
    }
    labels = {name: sides(values) for name, values in measured.items()}
    described = {
        name: properties.column(f"{name}_measured").to_numpy(zero_copy_only=False)
        for name in ("r_magnitude", "radius")
    }
    trained = training()
    estimated = {name: page_sides(measured[name], predicted[name]) for name in PAIRS}
    phrasings: dict[str, dict[str, Any]] = {name: {} for name in PAIRS}
    cross_pairs: dict[str, dict[str, Any]] = {}
    morphology: dict[str, dict[str, Any]] = {}
    for family, scorer, which in families(views):
        vectors = dict(zip(texts, queries[which]))
        edits = [edited(vectors[high], vectors[low], steps) for high, low in phrased]
        scored = scorer(queries[which], slice(None))
        nulls = scorer(np.concatenate([random_directions, *edits]), held)
        for name, scores in scored.items():
            method = f"{family}, {name}"
            by_text = dict(zip(texts, scores))
            morphology[method] = morphology_rows(
                by_text, reference, imaged, shapes, described, trained
            )
            cross_pairs[method] = {
                target: {
                    source: {
                        both[0]: {
                            "separation": separation(
                                by_text, both, labels[target], held
                            )
                        }
                        for both in found
                    }
                    for source, found in chosen.items()
                }
                for target in PAIRS
            }
            random_scores, *edit_scores = np.split(
                nulls[name], np.arange(RANDOM_DIRECTIONS, len(nulls[name]), 2 * EDITS)
            )
            for pair, found in chosen.items():
                phrasings[pair][method] = pair_rows(
                    by_text,
                    dict(zip(phrased, edit_scores)),
                    random_scores,
                    found,
                    labels[pair],
                    estimated[pair],
                    held,
                    reference,
                    imaged,
                )
    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "half": half,
        "galaxies": len(held),
        "random_directions": RANDOM_DIRECTIONS,
        "edits": EDITS,
        "pairs": {
            name: {
                "label": pair.label,
                "mode": pair.mode,
                "galaxies": int(labels[name]["known"][held].sum()),
            }
            | {
                f"{side}_base_rate": round(
                    float(labels[name][side][held][labels[name]["known"][held]].mean()),
                    4,
                )
                for side in SIDES
            }
            for name, pair in PAIRS.items()
        },
        "phrasings": phrasings,
        "cross_pairs": cross_pairs,
        "morphology": morphology,
        "memory": memory(),
    }


@app.local_entrypoint()
def main(half: str = HALVES[0]) -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **benchmark_text_search_quality.remote(half),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
