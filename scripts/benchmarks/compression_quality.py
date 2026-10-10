import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow.compute as pc
import torch

from app import predictions
from app.config import DATASET_REVISION, SEED, STORE_COLUMNS, galaxy_count
from app.dataset import spectrum
from app.search import source
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks import compression
from scripts.benchmarks.common import environment, git, memory, observed_spans
from scripts.benchmarks.pql_quality import redshifts

app = modal.App("alphauniverse-compression-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/compression_quality.json")
TOPS = (10, 32)
WINDOW = 16
TIMED = 16
SLOT_CHUNK = 32
GALAXY_CHUNK = 16
DIFFERENCES = 1_000_000
CHECK = 64
BUDGETS = (1e9, 2e9, 4e9, 8e9)
TARGETS = {
    "spearman_median": 0.99,
    "spearman_p5": 0.95,
    "top10_median": 0.9,
    "identity": 0.005,
    "relevance": 0.005,
}

Mode = predictions.Mode


def kind(mode: Mode) -> str:
    return mode.name.split("_")[-1]


def slot_count(mode: Mode) -> int:
    return len(mode.keys) * len(mode.positions)


def observed(survey: str) -> np.ndarray:
    column = source("tokens").to_table(columns=[survey]).column(survey)
    return pc.is_valid(column).to_numpy(zero_copy_only=False)


def draw(
    rng: np.random.Generator, sample: int, fit: int
) -> tuple[np.ndarray, np.ndarray]:
    sdss = observed("sdss")
    paired = (observed("hsc") | observed("desi")) & ~sdss
    chosen = np.flatnonzero(sdss)[:sample]
    wanted = min(max(sample // 2 - len(chosen), 0), int(paired.sum()))
    chosen = np.concatenate(
        (chosen, rng.choice(np.flatnonzero(paired), wanted, replace=False))
    )
    rest = np.setdiff1d(np.arange(galaxy_count()), chosen)
    chosen = np.concatenate(
        (chosen, rng.choice(rest, sample - len(chosen), replace=False))
    )
    remaining = np.setdiff1d(rest, chosen)
    return np.sort(chosen), np.sort(
        rng.choice(remaining, min(fit, len(remaining)), replace=False)
    )


def predict(
    galaxies: np.ndarray,
    modes: tuple[Mode, ...],
    hidden: bool = False,
    check: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    table = source("tokens").to_table(columns=["galaxy", *STORE_COLUMNS])
    found = {
        mode.name: np.empty(
            (len(galaxies), slot_count(mode), mode.vocabulary),
            dtype=np.float16 if kind(mode) == "image" else np.float32,
        )
        for mode in modes
    }
    exact = {
        mode.name: np.empty(
            (min(check, len(galaxies)), slot_count(mode), mode.vocabulary),
            dtype=np.float32,
        )
        for mode in modes
        if kind(mode) == "image"
    }
    targets = {key: mode.positions for mode in modes for key in mode.keys}
    for position, row in enumerate(table.take(galaxies).to_pylist()):
        tokens = predictions.inputs(row)
        if hidden:
            for mode in modes:
                for key in mode.keys:
                    tokens.pop(key, None)
        predicted = predictions.predictions(tokens, targets)
        for mode in modes:
            values = predictions.distributions(predicted, mode)
            found[mode.name][position] = values
            if mode.name in exact and position < len(exact[mode.name]):
                exact[mode.name][position] = values
    return found, exact


def selections(
    mode: Mode, galaxy: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    slots = slot_count(mode)
    if kind(mode) == "image":
        side = int(np.sqrt(slots))
        grid = np.arange(slots).reshape(side, side)
        blocks = {}
        for width in (4, 8):
            row, column = rng.integers(side - width + 1, size=2)
            blocks[width] = grid[row : row + width, column : column + width].ravel()
        return {
            "1": rng.choice(slots, 1),
            "16": blocks[4],
            "64": blocks[8],
            "all": np.arange(slots),
        }
    if kind(mode) == "spectrum":
        wavelength = spectrum(galaxy, mode.survey).column("wavelength").to_numpy()
        seen = observed_spans(wavelength)
        start = int(rng.integers(max(len(seen) - WINDOW, 0) + 1))
        return {
            "1": rng.choice(seen, 1),
            "16": seen[start : start + WINDOW],
            "all": seen,
        }
    if kind(mode) == "table":
        return {
            "1": rng.choice(slots, 1),
            "4": rng.choice(slots, 4, replace=False),
            "all": np.arange(slots),
        }
    return {"1": np.arange(slots)}


def chunks(
    gallery: np.ndarray, hidden: np.ndarray, device: torch.device
) -> Iterator[tuple[slice, torch.Tensor, torch.Tensor]]:
    for start in range(0, gallery.shape[1], SLOT_CHUNK):
        part = slice(start, start + SLOT_CHUNK)
        yield (
            part,
            torch.from_numpy(gallery[:, part]).to(device),
            torch.from_numpy(hidden[:, part]).to(device),
        )


def log_overlaps(
    scheme: Any,
    mode: Mode,
    gallery: np.ndarray,
    queries: np.ndarray,
    hidden: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    sides = ("exact",) if scheme is None else ("gallery", "both")
    shape = (gallery.shape[1], len(queries))
    found = {side: torch.empty(*shape, len(gallery), device=device) for side in sides}
    pairs = {side: torch.empty(*shape, device=device) for side in sides}
    bits = 0
    for part, rows, unseen in chunks(gallery, hidden, device):
        query = rows[queries].float().exp()
        if scheme is None:
            decoded, decoded_unseen = rows.float().exp(), unseen.float().exp()
            forms = {"exact": query}
        else:
            stored = scheme.encode(rows)
            bits += scheme.bits(stored)
            decoded = scheme.decode(stored)
            decoded_unseen = scheme.decode(scheme.encode(unseen))
            compressed = scheme.decode(scheme.encode(rows[queries]), side="query")
            forms = {"gallery": query, "both": compressed}
        for side, form in forms.items():
            found[side][part] = torch.einsum("qsv,gsv->sqg", form, decoded)
            pairs[side][part] = torch.einsum("qsv,qsv->sq", form, decoded_unseen)
    return (
        {side: logged(values, mode) for side, values in found.items()},
        {side: logged(values, mode) for side, values in pairs.items()},
        bits,
    )


def logged(overlaps: torch.Tensor, mode: Mode) -> torch.Tensor:
    return compression.floor(overlaps, mode.vocabulary).log()


def ranks(scores: torch.Tensor) -> torch.Tensor:
    return scores.argsort(dim=-1).argsort(dim=-1).float()


def spearman(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first, second = ranks(first), ranks(second)
    first = first - first.mean(dim=-1, keepdim=True)
    second = second - second.mean(dim=-1, keepdim=True)
    return (first * second).sum(dim=-1) / (first.norm(dim=-1) * second.norm(dim=-1))


def chosen(scores: torch.Tensor, top: int) -> torch.Tensor:
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter_(
        -1, scores.topk(min(top, scores.shape[-1]), dim=-1).indices, True
    )


def relevance(
    scores: torch.Tensor,
    others: torch.Tensor,
    redshift: torch.Tensor,
    own: torch.Tensor,
) -> torch.Tensor:
    gallery = redshift[others]
    known = torch.isfinite(gallery)
    top = scores.masked_fill(~known, -torch.inf)
    top = top.topk(min(TOPS[0], top.shape[-1]), dim=-1).indices
    offsets = (gallery.gather(-1, top) - own[:, None]).abs() / (1 + own[:, None])
    return offsets.nanmedian(dim=-1).values


def summary(
    found: torch.Tensor,
    pairs: torch.Tensor,
    reference: torch.Tensor,
    reference_pairs: torch.Tensor,
    masks: torch.Tensor,
    positions: torch.Tensor,
    redshift: torch.Tensor,
) -> dict[str, Any]:
    weights = masks.float()
    scores = torch.einsum("sqg,qs->qg", found, weights)
    exact = torch.einsum("sqg,qs->qg", reference, weights)
    galaxies = scores.shape[-1]
    everyone = torch.arange(galaxies, device=scores.device).expand(len(positions), -1)
    others = everyone[everyone != positions[:, None]].reshape(len(positions), -1)
    scores, exact = scores.gather(-1, others), exact.gather(-1, others)
    correlation = spearman(scores, exact)
    unseen = torch.einsum("sq,qs->q", pairs, weights)
    unseen_exact = torch.einsum("sq,qs->q", reference_pairs, weights)
    own = redshift[positions]
    known = torch.isfinite(own)
    found_relevance = relevance(scores, others, redshift, own)[known]
    exact_relevance = relevance(exact, others, redshift, own)[known]
    identity = (unseen > scores.amax(dim=-1)).float().mean()
    identity_exact = (unseen_exact > exact.amax(dim=-1)).float().mean()
    result = {
        "spearman_median": float(correlation.median()),
        "spearman_p5": float(correlation.quantile(0.05)),
        "spearman_min": float(correlation.min()),
        "identity": float(identity),
        "identity_exact": float(identity_exact),
        "relevance": float(found_relevance.nanmedian()) if known.any() else None,
        "relevance_exact": float(exact_relevance.nanmedian()) if known.any() else None,
    }
    for top in TOPS:
        shared = (chosen(scores, top) & chosen(exact, top)).sum(dim=-1)
        result[f"top{top}_median"] = float((shared / min(top, galaxies - 1)).median())
    return result


def slot_errors(found: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    differences = (found - reference).abs().reshape(-1)
    generator = torch.Generator(device=differences.device).manual_seed(SEED)
    picked = torch.randint(
        len(differences),
        (min(DIFFERENCES, len(differences)),),
        generator=generator,
        device=differences.device,
    )
    sampled = differences[picked]
    return {
        "median": float(sampled.median()),
        "p99": float(sampled.quantile(0.99)),
    }


def scan_ms(scheme: Any, gallery: np.ndarray, query: np.ndarray) -> float:
    rows = torch.from_numpy(gallery[:, :TIMED]).float()
    stored = scheme.encode(rows)
    form = torch.from_numpy(query[:TIMED]).float().exp()
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        scheme.overlap(stored, form)
        timings.append((time.perf_counter() - start) * 1000)
    return round(min(timings), 3)


def meets(entry: dict[str, Any]) -> bool:
    for result in entry["sizes"].values():
        both = result["both"]
        if (
            both["spearman_median"] < TARGETS["spearman_median"]
            or both["spearman_p5"] < TARGETS["spearman_p5"]
            or both["top10_median"] < TARGETS["top10_median"]
            or abs(both["identity"] - both["identity_exact"]) > TARGETS["identity"]
        ):
            return False
        if both["relevance"] is not None and (
            abs(both["relevance"] - both["relevance_exact"]) > TARGETS["relevance"]
        ):
            return False
    return True


def worst(entry: dict[str, Any]) -> float:
    return min(result["both"]["spearman_median"] for result in entry["sizes"].values())


def choices(results: dict[str, dict[str, Any]], total: int) -> dict[str, Any]:
    accurate = {}
    for mode, entries in results.items():
        passing = [name for name, entry in entries.items() if meets(entry)]
        accurate[mode] = min(
            passing, key=lambda name: entries[name]["bytes_per_galaxy"], default=None
        )
    thresholds = sorted(
        {worst(entry) for entries in results.values() for entry in entries.values()},
        reverse=True,
    )
    budgets = {}
    for budget in BUDGETS:
        budgets[f"{budget / 1e9:g}_gb"] = None
        for threshold in thresholds:
            picks = {}
            for mode, entries in results.items():
                eligible = [
                    name for name, entry in entries.items() if worst(entry) >= threshold
                ]
                if eligible:
                    picks[mode] = min(
                        eligible, key=lambda name: entries[name]["bytes_per_galaxy"]
                    )
            if len(picks) < len(results):
                continue
            size = (
                sum(
                    results[mode][name]["bytes_per_galaxy"]
                    for mode, name in picks.items()
                )
                * total
            )
            if size <= budget:
                budgets[f"{budget / 1e9:g}_gb"] = {
                    "worst_spearman_median": threshold,
                    "corpus_bytes": size,
                    "schemes": picks,
                }
                break
    return {"accuracy": accurate, "budgets": budgets}


@app.function(
    image=image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=24 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def benchmark_compression_quality(
    sample: int, fit: int, queries: int
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    galaxies, fitting = draw(rng, sample, fit)
    started = time.perf_counter()
    evaluated, exact = predict(galaxies, predictions.MODES, check=CHECK)
    fitted, _ = predict(fitting, predictions.MODES)
    prediction_s = time.perf_counter() - started
    redshift = torch.from_numpy(redshifts()[galaxies]).float().to(device)
    total = galaxy_count()

    results: dict[str, dict[str, Any]] = {}
    checks = {}
    for mode in predictions.MODES:
        seen = np.flatnonzero(observed(mode.survey)[galaxies])
        picked = np.sort(rng.choice(seen, min(queries, len(seen)), replace=False))
        if not len(picked):
            continue
        chosen_slots = [
            selections(mode, int(galaxies[position]), rng) for position in picked
        ]
        sizes = list(chosen_slots[0])
        masks = {}
        for size in sizes:
            mask = torch.zeros(len(picked), slot_count(mode), dtype=torch.bool)
            for row, selection in enumerate(chosen_slots):
                mask[row, torch.from_numpy(np.asarray(selection[size]))] = True
            masks[size] = mask.to(device)
        positions = torch.from_numpy(picked).to(device)
        hidden, _ = predict(galaxies[picked], (mode,), hidden=True)
        gallery, unseen = evaluated[mode.name], hidden[mode.name]

        basis = compression.Basis(max(compression.RANKS))
        basis.fit(
            torch.from_numpy(fitted[mode.name][start : start + GALAXY_CHUNK]).to(device)
            for start in range(0, len(fitting), GALAXY_CHUNK)
        )
        reference, reference_pairs, _ = log_overlaps(
            None, mode, gallery, picked, unseen, device
        )
        if mode.name in exact:
            full = torch.from_numpy(exact[mode.name]).to(device).float().exp()
            half = torch.from_numpy(gallery[: len(full)]).to(device).float().exp()
            differences = (
                torch.einsum("asv,bsv->sab", half, half).log()
                - torch.einsum("asv,bsv->sab", full, full).log()
            ).abs()
            checks[mode.name] = {
                "galaxies": len(full),
                "median": float(differences.median()),
                "max": float(differences.max()),
            }
            del full, half, differences

        entries = {}
        for scheme in compression.schemes(mode, basis):
            found, pairs, bits = log_overlaps(
                scheme, mode, gallery, picked, unseen, device
            )
            entries[scheme.name] = {
                "bytes_per_galaxy": bits / 8 / len(gallery),
                "corpus_gb": round(bits / 8 / len(gallery) * total / 1e9, 4),
                "scan_ms": scan_ms(scheme, gallery, gallery[picked[0]]),
                "slot_error": {
                    side: slot_errors(found[side], reference["exact"]) for side in found
                },
                "sizes": {
                    size: {
                        side: summary(
                            found[side],
                            pairs[side],
                            reference["exact"],
                            reference_pairs["exact"],
                            masks[size],
                            positions,
                            redshift,
                        )
                        for side in found
                    }
                    for size in sizes
                },
            }
            del found, pairs
        results[mode.name] = entries

    return {
        "environment": environment(),
        "device": str(device),
        "sample": len(galaxies),
        "fit": len(fitting),
        "queries": queries,
        "prediction_s": round(prediction_s, 1),
        "targets": TARGETS,
        "fp16_reference_check": checks,
        "results": results,
        "choices": choices(results, total),
        "memory": memory(),
    }


@app.local_entrypoint()
def main(sample: int = 1024, fit: int = 512, queries: int = 200) -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **benchmark_compression_quality.remote(sample, fit, queries),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
