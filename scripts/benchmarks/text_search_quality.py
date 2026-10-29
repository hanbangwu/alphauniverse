import json
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path
from typing import Any, NamedTuple

import modal
import numpy as np
import pyarrow.parquet as pq
import torch
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
from app.embeddings import loaded
from app.text_flow import generated_clouds, stored_flow
from app.text_search import (
    FRAMES,
    TextIndex,
    aion_gemma_space,
    embed_queries,
    generic_queries,
    indexed,
    likelihoods,
    text_index,
    view_kinds,
)
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
HALVES = ("development", "sealed")
SINGLE = ("images", "spectra", "catalogues")
MODE_SETS = {"all": MODES} | {name: VIEW_COMBINATIONS[name] for name in SINGLE}
FRAMED = dict(zip(SINGLE, FRAMES[1:]))
MODALITIES = {
    "all": tuple(TEXT_SURVEYS),
    "images": ("image",),
    "spectra": ("spectrum",),
}
RANDOM_DIRECTIONS = 200
EDITS = 50
TOP = 32
BATCH = 64
REPORT = Path("docs/benchmarks/text_search_quality.json")

type Scorer = Callable[[np.ndarray], np.ndarray]


class Method(NamedTuple):
    score: Scorer
    again: Scorer | None
    queries: str


def centred(galaxy: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return galaxy, normalize(vectors - vectors.mean(axis=0))


def view_scorer(
    found: dict[str, tuple[np.ndarray, np.ndarray]], modalities: tuple[str, ...]
) -> Scorer:
    count = np.zeros(galaxy_count())
    for modality in modalities:
        count[found[modality][0]] += 1

    def score(directions: np.ndarray) -> np.ndarray:
        total = np.zeros((len(directions), galaxy_count()))
        for modality in modalities:
            galaxy, vectors = found[modality]
            total[:, galaxy] += directions @ vectors.T
        return np.where(count > 0, total / np.maximum(count, 1), -1.0)

    return score


def likelihood_scorer(index: TextIndex, kinds: list[int]) -> Scorer:
    def score(directions: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                likelihoods(index, torch.from_numpy(np.stack(rows)).float(), kinds)[
                    0
                ].numpy()
                for rows in batched(directions, BATCH)
            ]
        )

    return score


def summaries(index: TextIndex) -> torch.Tensor:
    found = torch.empty(*index.points.shape[:2], index.points.shape[-1] + 1)
    for kind, (points, lengths) in enumerate(zip(index.points, index.lengths)):
        inverse = 1 / lengths
        found[kind, :, :-1] = (points * inverse[..., None]).mean(dim=1)
        found[kind, :, -1] = inverse.mean(dim=1)
    return found


def mean_scorer(index: TextIndex, summary: torch.Tensor, kinds: list[int]) -> Scorer:
    def score(directions: np.ndarray) -> np.ndarray:
        chosen = torch.from_numpy(directions).float().T
        total = sum(
            summary[kind, :, :-1] @ (index.components[kind] @ chosen)
            + summary[kind, :, -1:] * (index.centre[kind] @ chosen)
            for kind in kinds
        )
        return (total / len(kinds)).T.numpy()

    return score


def methods(
    index: TextIndex, second: TextIndex, views: dict, maps: dict
) -> dict[str, Method]:
    summary, again = summaries(index), summaries(second)
    found = {}
    for name, modes in MODE_SETS.items():
        kinds = view_kinds(modes)
        found[f"clouds, {name}"] = Method(
            likelihood_scorer(index, kinds), likelihood_scorer(second, kinds), "plain"
        )
        averaged = Method(
            mean_scorer(index, summary, kinds),
            mean_scorer(second, again, kinds),
            "plain",
        )
        found[f"mean cosine, {name}"] = averaged
        if name in FRAMED:
            found[f"mean cosine framed, {name}"] = averaged._replace(queries=name)
        if name in MODALITIES:
            for method, space in (("views", views), ("maps", maps)):
                found[f"{method}, {name}"] = Method(
                    view_scorer(space, MODALITIES[name]), None, "plain"
                )
    return found


def sides(values: np.ndarray) -> dict[str, np.ndarray]:
    low, high = np.nanquantile(values, [1 / 3, 2 / 3])
    return {"high": values > high, "low": values < low, "known": np.isfinite(values)}


def edited(
    rng: np.random.Generator, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    middle = (first + second) / 2
    steps = normalize(rng.standard_normal((EDITS, len(first))))
    steps *= np.linalg.norm(first - second) / 2
    return np.concatenate([normalize(middle + steps), normalize(middle - steps)])


def top(score: np.ndarray, kept: np.ndarray) -> np.ndarray:
    order = np.argsort(-score)
    return order[kept[order]][:TOP]


def pair_rows(
    method: Method,
    vectors: dict[str, np.ndarray],
    edits: dict[tuple[str, str], np.ndarray],
    phrasings: tuple[tuple[str, str], ...],
    labels: dict[str, np.ndarray],
    held: np.ndarray,
    random_scores: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    known = labels["known"]
    tested = held[known[held]]

    def precision(relevant: np.ndarray, score: np.ndarray) -> float:
        return float(average_precision_score(relevant[tested], score[tested]))

    nulls = {
        side: np.quantile([precision(labels[side], row) for row in random_scores], 0.95)
        for side in ("high", "low")
    }
    rows: dict[str, list[dict[str, Any]]] = {"high": [], "low": []}
    for high, low in phrasings:
        scores = dict(
            zip((high, low), method.score(np.stack([vectors[high], vectors[low]])))
        )
        plus, minus = np.split(method.score(edits[high, low]), 2)
        for side, own, opposite in (("high", high, low), ("low", low, high)):
            relevant, other = labels[side], labels["low" if side == "high" else "high"]
            own_ap = precision(relevant, scores[own])
            opposite_ap = precision(relevant, scores[opposite])
            edit_null = np.quantile(
                [
                    abs(precision(relevant, a) - precision(relevant, b))
                    for a, b in zip(plus, minus)
                ],
                0.95,
            )
            page = top(scores[own], known)
            row = {
                "phrasing": own,
                "average_precision": round(own_ap, 4),
                "opposite_average_precision": round(opposite_ap, 4),
                "specificity": round(own_ap - opposite_ap, 4),
                "edit_null": round(float(edit_null), 4),
                "random_null": round(float(nulls[side]), 4),
                "passes": bool(
                    own_ap - opposite_ap > edit_null and own_ap > nulls[side]
                ),
                f"precision@{TOP}": round(float(relevant[page].mean()), 4),
                f"opposite_precision@{TOP}": round(float(other[page].mean()), 4),
            }
            if method.again is not None:
                everyone = np.ones_like(known)
                first = top(scores[own], everyone)
                second = top(method.again(vectors[own][None])[0], everyone)
                row[f"seed_overlap@{TOP}"] = len(np.intersect1d(first, second)) / TOP
            rows[side].append(row)
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
    texts = [text for found in chosen.values() for both in found for text in both]

    index = text_index()
    bank = embed_queries(generic_queries())
    frame_means = dict(zip(FRAMES, np.split(bank, len(FRAMES))))
    queries = {"plain": normalize(embed_queries(texts) - index.query_mean.numpy())}
    for name, frame in FRAMED.items():
        framed = embed_queries([frame.format(text) for text in texts])
        queries[name] = normalize(framed - frame_means[frame].mean(axis=0))
    queries = {name: dict(zip(texts, rows)) for name, rows in queries.items()}
    rng = np.random.default_rng(SEED)
    edits = {
        name: {
            both: edited(rng, vectors[both[0]], vectors[both[1]])
            for found in chosen.values()
            for both in found
        }
        for name, vectors in queries.items()
    }
    random_directions = normalize(
        rng.standard_normal((RANDOM_DIRECTIONS, bank.shape[1]))
    )

    views = {}
    for modality in TEXT_SURVEYS:
        galaxy, _, gemma = pairs(modality)
        views[modality] = centred(galaxy, gemma.cpu().numpy())
    space = aion_gemma_space()
    maps = {
        modality: centred(
            space.galaxy[space.modality == modality],
            space.vectors[space.modality == modality],
        )
        for modality in TEXT_SURVEYS
    }
    network, bases = stored_flow()
    joint = loaded("joint_embedding", torch.float32)
    every = torch.arange(galaxy_count())
    second = indexed(generated_clouds(network, bases, joint, every, SEED + 1))

    properties = pq.read_table(artifact("properties"))
    labels = {
        name: sides(properties.column(pair.label).to_numpy(zero_copy_only=False))
        for name, pair in PAIRS.items()
    }
    results: dict[str, Any] = {}
    for name, pair in PAIRS.items():
        tested = held[labels[name]["known"][held]]
        results[name] = {
            "label": pair.label,
            "mode": pair.mode,
            "galaxies": len(tested),
            "base_rates": {
                side: round(float(labels[name][side][tested].mean()), 4)
                for side in ("high", "low")
            },
            "methods": {},
        }
    for method_name, method in methods(index, second, views, maps).items():
        random_scores = method.score(random_directions)
        for name in PAIRS:
            results[name]["methods"][method_name] = pair_rows(
                method,
                queries[method.queries],
                edits[method.queries],
                chosen[name],
                labels[name],
                held,
                random_scores,
            )
    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "half": half,
        "galaxies": len(held),
        "kappa": index.kappa,
        "generic_queries": len(bank),
        "random_directions": RANDOM_DIRECTIONS,
        "edits": EDITS,
        "pairs": results,
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
