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
    alignment,
    embeddings,
    encode,
    grams,
    observed_grams,
    pql,
    predictions,
    templates,
)
from app.config import (
    DATASET_REVISION,
    GRAM_NAMES,
    STORE_COLUMNS,
    TOKEN_GRAMS,
    artifact,
    device,
)
from app.search import source
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-retrieval-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/retrieval_quality.json")
SAMPLE = 512
SEED = 0
BATCH = 256
RANKS = (64, 256, 1024)
TESTS = {
    "full_context": (("ls",), STORE_COLUMNS),
    "ls_alone": (("ls", "hsc"), ("ls",)),
    "hsc_alone": (("ls", "hsc"), ("hsc",)),
    "desi_alone": (("desi", "sdss"), ("desi",)),
    "sdss_alone": (("desi", "sdss"), ("sdss",)),
    "hidden_spectrum": (("desi",), ("ls", "hsc", "sdss")),
}
COMBINED = {
    "same_place": tuple(TOKEN_GRAMS),
    "any_position": tuple(names[-1] for names in TOKEN_GRAMS.values()),
}


def chosen(store: pa.Table, required: tuple[str, ...]) -> np.ndarray:
    present = np.logical_and.reduce(
        [
            store.column(survey).is_valid().to_numpy(zero_copy_only=False)
            for survey in required
        ]
    )
    galaxies = np.flatnonzero(present)
    if len(galaxies) <= SAMPLE:
        return galaxies
    return np.sort(np.random.default_rng(SEED).choice(galaxies, SAMPLE, replace=False))


@torch.inference_mode()
def encoded(rows: list[dict], kept: tuple[str, ...]) -> tuple[pa.Table, torch.Tensor]:
    fitted = {survey: basis[:2] for survey, basis in pql.basis().items()}
    records, means = [], []
    for row in tqdm(rows, desc="encode"):
        hidden = {survey: None for survey in STORE_COLUMNS if survey not in kept}
        context, _, mask, _ = encode.context(predictions.inputs(row | hidden))
        means.append(context[0].float().mean(dim=0))
        predicted = predictions.decode(context, mask, predictions.TARGETS)
        records.append(predictions.record(row["galaxy"], predicted, fitted))
    return pa.Table.from_batches([pql.prediction_batch(records)]), torch.stack(means)


def token_rows(tests: dict[str, pa.Table]) -> dict[str, dict[str, torch.Tensor]]:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    found = {test: {} for test in tests}
    for mode in tqdm(TOKEN_GRAMS, desc="token rows"):
        gallery = grams.forms(mode, pql.predictions())
        for test, table in tests.items():
            queries = grams.forms(mode, table)
            found[test] |= grams.rows(mode, queries, table.num_rows, gallery)
    return found


def template_rows(
    tests: dict[str, pa.Table], references: dict[str, np.ndarray]
) -> dict[str, dict[str, torch.Tensor]]:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    decoded = {}
    for test, table in tests.items():
        parts = [templates.templates(rows) for rows in table.to_batches(BATCH)]
        decoded[test] = {
            mode: np.concatenate([part[mode] for part in parts]) for mode in parts[0]
        }
    found = {test: {} for test in tests}
    for mode, names in tqdm(TOKEN_GRAMS.items(), desc="template rows"):
        depth = torch.from_numpy(references[f"{mode}_depth"]).to(device())
        everyone = embeddings.loaded(f"template_{mode}", torch.float32).flatten(1)
        gallery = observed_grams.at_depth(everyone, depth, mode)
        if len(names) > 1:
            kept = observed_grams.slotted(depth[None], mode)[0].sum(-1) > 0
            right = observed_grams.template_features(gallery, mode, kept)[1]
        for test in tests:
            values = torch.from_numpy(decoded[test][mode]).to(device()).flatten(1)
            queries = observed_grams.at_depth(values, depth, mode)
            found[test][names[0]] = observed_grams.template_gram(
                queries, gallery
            ).float()
            if len(names) > 1:
                found[test][names[1]] = observed_grams.anywhere(
                    observed_grams.template_features(queries, mode, kept)[0],
                    right,
                    symmetric=False,
                )
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


def standardised(scores: torch.Tensor) -> torch.Tensor:
    return (scores - scores.mean(dim=1, keepdim=True)) / scores.std(dim=1, keepdim=True)


def embedded(role: str) -> dict[str, dict[str, torch.Tensor]]:
    found = {}
    for name in GRAM_NAMES:
        with np.load(artifact(f"{role}_{name}")) as stored:
            found[name] = {
                part: torch.from_numpy(stored[part]).to(device())
                for part in ("galleries", "offsets", "singular_values")
            }
    return found


def family(
    found: dict[str, dict[str, torch.Tensor]],
    rows: dict[str, torch.Tensor],
    sources: torch.Tensor,
) -> dict[str, Any]:
    scored = {"exact": rows} | {
        rank: {
            name: embeddings.scores(row, found[name], rank)
            for name, row in rows.items()
        }
        for rank in RANKS
    }
    report = {
        version: {name: recall(ranks(row, sources)) for name, row in blocks.items()}
        for version, blocks in scored.items()
    }
    for version, blocks in scored.items():
        for combination, names in COMBINED.items():
            total = sum(standardised(blocks[name]) for name in names)
            report[version][combination] = recall(ranks(total, sources))
    return report


@app.function(
    image=image,
    gpu="H100!",
    cpu=4,
    memory=(16 * 1024, 64 * 1024),
    timeout=75 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    store = source("tokens").to_table(columns=["galaxy", *STORE_COLUMNS])
    galaxies, tests, means = {}, {}, {}
    for test, (required, kept) in TESTS.items():
        galaxies[test] = chosen(store, required)
        rows = store.take(galaxies[test]).to_pylist()
        tests[test], means[test] = encoded(rows, kept)
    with np.load(artifact("observed_reference")) as stored:
        references = dict(stored)
    rows = {"token": token_rows(tests), "template": template_rows(tests, references)}
    stored = {role: embedded(f"{role}_embedding") for role in rows}
    _, aion, _ = alignment.pairs()
    aion = torch.nn.functional.normalize(aion, dim=1)
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
        "galaxies": store.num_rows,
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
