import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from app import (
    embeddings,
    encode,
    grams,
    observed_grams,
    parametric_umap,
    pql,
    predictions,
    templates,
)
from app.config import (
    DATASET_REVISION,
    GRAM_NAMES,
    SEED,
    SPECTRUM_SURVEYS,
    STORE_COLUMNS,
    TOKEN_EMBEDDING_RANK,
    TOKEN_GRAMS,
    artifact,
    device,
    galaxy_count,
)
from app.search import tokens
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import (
    COMBINED,
    environment,
    git,
    memory,
    standardised,
)

app = modal.App("alphauniverse-retrieval-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/retrieval_quality.json")
SAMPLE = 512
BATCH = 256
RANKS = tuple(TOKEN_EMBEDDING_RANK // 4**quarterings for quarterings in (2, 1, 0))
TESTS = {
    "full_context": (("ls",), STORE_COLUMNS),
    "ls_alone": (("ls", "hsc"), ("ls",)),
    "hsc_alone": (("ls", "hsc"), ("hsc",)),
    "desi_alone": (("desi", "sdss"), ("desi",)),
    "sdss_alone": (("desi", "sdss"), ("sdss",)),
    "hidden_spectrum": (("desi",), ("ls", "hsc", "sdss")),
}


def chosen(required: tuple[str, ...]) -> np.ndarray:
    galaxies = np.flatnonzero(
        np.logical_and.reduce([pql.observed(survey) for survey in required])
    )
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(galaxies, min(SAMPLE, len(galaxies)), replace=False))


def encoded(
    galaxies: np.ndarray, kept: tuple[str, ...]
) -> tuple[pa.Table, torch.Tensor]:
    with np.load(artifact("probability_basis")) as stored:
        fitted = {
            survey: (stored[f"{survey}_mean"], stored[f"{survey}_directions"])
            for survey in SPECTRUM_SURVEYS
        }
    hidden = dict.fromkeys(column for column in STORE_COLUMNS if column not in kept)
    records, means = [], []
    for row in tqdm(tokens().take(galaxies).to_pylist(), desc="encode"):
        context, _, mask, _ = encode.context(predictions.inputs(row | hidden))
        means.append(context[0].float().mean(dim=0))
        predicted = predictions.decode(context, mask, predictions.TARGETS)
        records.append(predictions.record(row["galaxy"], predicted, fitted))
    return pa.Table.from_batches([pql.prediction_batch(records)]), torch.stack(means)


def token_rows(tests: dict[str, pa.Table]) -> dict[str, dict[str, torch.Tensor]]:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    found = {test: {} for test in tests}
    for mode in TOKEN_GRAMS:
        gallery = grams.forms(mode, pql.predictions())
        for test, table in tests.items():
            queries = grams.forms(mode, table)
            found[test] |= grams.gram(mode, queries, gallery, symmetric=False)
    return found


def template_rows(tests: dict[str, pa.Table]) -> dict[str, dict[str, torch.Tensor]]:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    with np.load(artifact("observed_reference")) as references:
        depths = {
            mode: torch.from_numpy(references[f"{mode}_depth"]).to(device())
            for mode in TOKEN_GRAMS
        }
    decoded = {}
    for test, table in tests.items():
        parts = [templates.templates(rows) for rows in table.to_batches(BATCH)]
        decoded[test] = {
            mode: np.concatenate([part[mode] for part in parts]) for mode in parts[0]
        }
    found = {test: {} for test in tests}
    for mode in TOKEN_GRAMS:
        everyone = embeddings.loaded(f"template_{mode}", torch.float32).flatten(1)
        for test in tests:
            queries = torch.from_numpy(decoded[test][mode]).to(device()).flatten(1)
            matrices = observed_grams.template_grams(
                queries, depths[mode], mode, everyone
            )
            found[test] |= {name: matrix.float() for name, matrix in matrices.items()}
    return found


def embedded(role: str) -> dict[str, dict[str, torch.Tensor]]:
    found = {}
    for name in GRAM_NAMES:
        with np.load(artifact(f"{role}_{name}")) as stored:
            found[name] = {
                part: torch.from_numpy(stored[part]).to(device())
                for part in ("galleries", "offsets", "singular_values")
            }
    return found


def ranks(scores: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    own = scores.gather(1, sources[:, None])
    return (scores > own).sum(dim=1) + 1


def recall(found: torch.Tensor) -> dict[str, float]:
    return {
        "r1": round(float((found <= 1).double().mean()), 4),
        "r10": round(float((found <= 10).double().mean()), 4),
        "median": float(found.double().median()),
    }


def family(
    found: dict[str, dict[str, torch.Tensor]],
    rows: dict[str, torch.Tensor],
    sources: torch.Tensor,
) -> dict[str, Any]:
    vectors = {
        name: embeddings.query_vectors(row, found[name]) for name, row in rows.items()
    }
    scored = {"exact": rows} | {
        rank: {
            name: embeddings.scores(vectors[name], found[name], rank) for name in rows
        }
        for rank in RANKS
    }
    return {
        version: {
            "blocks": {
                name: recall(ranks(row, sources)) for name, row in blocks.items()
            },
            "combined": {
                combination: recall(
                    ranks(sum(standardised(blocks[name]) for name in names), sources)
                )
                for combination, names in COMBINED.items()
            },
        }
        for version, blocks in scored.items()
    }


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=(16 * 1024, 64 * 1024),
    timeout=75 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    galaxies = {test: chosen(required) for test, (required, _) in TESTS.items()}
    tests, means = {}, {}
    for test, (_, kept) in TESTS.items():
        tests[test], means[test] = encoded(galaxies[test], kept)
    rows = {"token": token_rows(tests), "template": template_rows(tests)}
    stored = {role: embedded(f"{role}_embedding") for role in rows}
    gallery = parametric_umap.scan(galaxy_count(), np.empty(0, dtype=np.intp))[0]
    aion = torch.nn.functional.normalize(torch.from_numpy(gallery).to(device()), dim=1)
    report = {}
    for test in tests:
        sources = torch.from_numpy(galaxies[test]).to(device())
        cosines = torch.nn.functional.normalize(means[test], dim=1) @ aion.T
        report[test] = {
            "galaxies": len(galaxies[test]),
            "kept": TESTS[test][1],
            "aion": recall(ranks(cosines, sources)),
        } | {role: family(stored[role], rows[role][test], sources) for role in rows}
    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "galaxies": pql.predictions().num_rows,
        "ranks": RANKS,
        "tests": report,
        "memory": memory(),
    }


@app.local_entrypoint()
def main() -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **measure.remote(),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
