import json
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import modal
import numpy as np
import torch

from app import pql, predictions
from app.config import DATASET_REVISION, SEED, STORE_COLUMNS, galaxy_count
from app.dataset import spectrum
from app.search import tokens
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks import compression
from scripts.benchmarks.common import (
    environment,
    git,
    memory,
    observed_spectrum_tokens,
)
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
kind = compression.kind


def slot_count(mode: Mode) -> int:
    return len(mode.keys) * len(mode.positions)


def draw(sample: int, fit: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    sdss = np.flatnonzero(pql.observed("sdss"))
    chosen = rng.choice(sdss, min(len(sdss), sample // 2), replace=False)
    paired = np.flatnonzero(pql.observed("hsc") | pql.observed("desi"))
    paired = np.setdiff1d(paired, chosen)
    wanted = min(max(sample // 2 - len(chosen), 0), len(paired))
    chosen = np.concatenate((chosen, rng.choice(paired, wanted, replace=False)))
    rest = np.setdiff1d(np.arange(galaxy_count()), chosen)
    chosen = np.concatenate(
        (chosen, rng.choice(rest, sample - len(chosen), replace=False))
    )
    remaining = np.setdiff1d(rest, chosen)
    fitting = rng.choice(remaining, min(fit, len(remaining)), replace=False)
    return np.sort(chosen), np.sort(fitting)


def predict(
    galaxies: np.ndarray, mode: Mode, hidden: bool = False, check: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rows = tokens().select(STORE_COLUMNS).take(galaxies).to_pylist()
    shape = (slot_count(mode), mode.vocabulary)
    found = np.empty(
        (len(galaxies), *shape),
        dtype=np.float16 if kind(mode) == "image" else np.float32,
    )
    exact = np.empty((min(check, len(galaxies)), *shape), dtype=np.float32)
    targets = {key: mode.positions for key in mode.keys}
    for position, row in enumerate(rows):
        context = predictions.inputs(row)
        if hidden:
            for key in mode.keys:
                context.pop(key, None)
        values = predictions.distributions(
            predictions.predictions(context, targets), mode
        )
        found[position] = values
        if position < len(exact):
            exact[position] = values
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
        seen = observed_spectrum_tokens(wavelength)
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


def logarithmic(scheme: Any, mode: Mode) -> bool:
    if kind(mode) in ("image", "spectrum"):
        return False
    return (
        scheme is None
        or isinstance(scheme, compression.Current)
        or (isinstance(scheme, compression.Dense) and scheme.precision.domain == "log")
    )


def exponentiated(log_probabilities: torch.Tensor) -> torch.Tensor:
    return log_probabilities.float().exp()


def shifted(log_probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    largest = log_probabilities.float().amax(dim=-1, keepdim=True)
    return (log_probabilities.float() - largest).double().exp(), largest[..., 0].T


def log_products(
    form: torch.Tensor, decoded: torch.Tensor, unseen: torch.Tensor, minimum: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    form_values, form_largest = shifted(form)
    decoded_values, decoded_largest = shifted(decoded)
    unseen_values, unseen_largest = shifted(unseen)
    overlaps = torch.einsum("qsv,gsv->sqg", form_values, decoded_values)
    own = torch.einsum("qsv,qsv->sq", form_values, unseen_values)
    return (
        overlaps.clamp_min(minimum).log()
        + form_largest[:, :, None]
        + decoded_largest[:, None, :],
        own.clamp_min(minimum).log() + form_largest + unseen_largest,
        int((overlaps < minimum).sum()),
    )


def products(
    form: torch.Tensor, decoded: torch.Tensor, unseen: torch.Tensor, minimum: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    overlaps = torch.einsum("qsv,gsv->sqg", form, decoded)
    own = torch.einsum("qsv,qsv->sq", form, unseen)
    return (
        overlaps.clamp_min(minimum).log(),
        own.clamp_min(minimum).log(),
        int((overlaps < minimum).sum()),
    )


def log_overlaps(
    scheme: Any,
    mode: Mode,
    gallery: np.ndarray,
    queries: np.ndarray,
    hidden: np.ndarray,
    device: torch.device,
    window: torch.Tensor | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    int,
    float,
]:
    sides = ("exact",) if scheme is None else ("gallery", "both")
    shape = (gallery.shape[1], len(queries))
    found = {side: torch.empty(*shape, len(gallery), device=device) for side in sides}
    pairs = {side: torch.empty(*shape, device=device) for side in sides}
    crossed: dict[str, torch.Tensor] = {}
    crossed_pairs: dict[str, torch.Tensor] = {}
    if logarithmic(scheme, mode):
        overlap, minimum = log_products, torch.finfo(torch.float64).tiny
        decode = compression.load
        query_decode = compression.load
        exact = torch.Tensor.float
    else:
        overlap, minimum = products, compression.floor(scheme, mode.vocabulary)
        decode = scheme and scheme.decode
        query_decode = scheme and partial(scheme.decode, side="query")
        exact = exponentiated
    if window is not None:
        selected = [row.nonzero()[:, 0] for row in window]
        targets = anywhere_forms(scheme, gallery, queries, selected, device)
        for side in sides:
            crossed[side] = torch.zeros(*shape, len(gallery), device=device)
            crossed_pairs[side] = torch.zeros(*shape, device=device)
            for position, slots in enumerate(selected):
                crossed[side][slots, position] = -torch.inf
                crossed_pairs[side][slots, position] = -torch.inf
    bits = floored = 0
    for part, rows, unseen in chunks(gallery, hidden, device):
        query = exact(rows[queries])
        if scheme is None:
            decoded, decoded_unseen = exact(rows), exact(unseen)
            forms = {"exact": query}
        else:
            stored = scheme.encode(rows)
            bits += scheme.bits(stored)
            decoded = decode(stored)
            decoded_unseen = decode(scheme.encode(unseen))
            forms = {
                "gallery": query,
                "both": query_decode(scheme.encode(rows[queries])),
            }
        for side, form in forms.items():
            found[side][part], pairs[side][part], below = overlap(
                form, decoded, decoded_unseen, minimum
            )
            floored += below
        for side, best in crossed.items():
            own_best = crossed_pairs[side]
            for position, (slots, form) in enumerate(
                zip(selected, targets[side], strict=True)
            ):
                overlaps = torch.einsum("kv,gcv->kgc", form, decoded)
                best[slots, position] = torch.maximum(
                    best[slots, position],
                    overlaps.clamp_min(minimum).log().amax(dim=-1),
                )
                own = torch.einsum("kv,cv->kc", form, decoded_unseen[position])
                own_best[slots, position] = torch.maximum(
                    own_best[slots, position],
                    own.clamp_min(minimum).log().amax(dim=-1),
                )
    total = sum(values.numel() for values in found.values())
    return found, pairs, crossed, crossed_pairs, bits, floored / total


def anywhere_forms(
    scheme: Any,
    gallery: np.ndarray,
    queries: np.ndarray,
    selected: list[torch.Tensor],
    device: torch.device,
) -> dict[str, list[torch.Tensor]]:
    rows = [
        torch.from_numpy(gallery[query][slots.cpu().numpy()]).to(device)[None]
        for query, slots in zip(queries, selected, strict=True)
    ]
    exact = [exponentiated(own)[0] for own in rows]
    if scheme is None:
        return {"exact": exact}
    return {
        "gallery": exact,
        "both": [scheme.decode(scheme.encode(own), side="query")[0] for own in rows],
    }


def ranks(scores: torch.Tensor) -> torch.Tensor:
    ordered = scores.sort(dim=-1).values.contiguous()
    values = scores.contiguous()
    low = torch.searchsorted(ordered, values)
    high = torch.searchsorted(ordered, values, right=True)
    return (low + high - 1).float() / 2


def spearman(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first, second = ranks(first), ranks(second)
    first = first - first.mean(dim=-1, keepdim=True)
    second = second - second.mean(dim=-1, keepdim=True)
    return (first * second).sum(dim=-1) / (first.norm(dim=-1) * second.norm(dim=-1))


def chosen(scores: torch.Tensor, top: int) -> torch.Tensor:
    mask = torch.zeros_like(scores, dtype=torch.bool)
    picked = scores.topk(min(top, scores.shape[-1]), dim=-1).indices
    return mask.scatter_(-1, picked, True)


def relevance(
    scores: torch.Tensor,
    others: torch.Tensor,
    redshift: torch.Tensor,
    own: torch.Tensor,
) -> torch.Tensor:
    gallery = redshift[others]
    ranked = scores.masked_fill(~torch.isfinite(gallery), -torch.inf)
    top = ranked.topk(min(TOPS[0], ranked.shape[-1]), dim=-1).indices
    offsets = (gallery.gather(-1, top) - own[:, None]).abs() / (1 + own[:, None])
    return offsets.nanmedian(dim=-1).values


def summary(
    found: torch.Tensor,
    pairs: torch.Tensor,
    reference: torch.Tensor,
    reference_pairs: torch.Tensor,
    mask: torch.Tensor,
    positions: torch.Tensor,
    redshift: torch.Tensor,
) -> dict[str, Any]:
    weights = mask.float()
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
    result = {
        "spearman_median": float(correlation.median()),
        "spearman_p5": float(correlation.quantile(0.05)),
        "spearman_min": float(correlation.min()),
        "identity": float((unseen > scores.amax(dim=-1)).float().mean()),
        "identity_exact": float((unseen_exact > exact.amax(dim=-1)).float().mean()),
        "relevance": None,
        "relevance_exact": None,
    }
    if known.any():
        result["relevance"] = float(
            relevance(scores, others, redshift, own)[known].nanmedian()
        )
        result["relevance_exact"] = float(
            relevance(exact, others, redshift, own)[known].nanmedian()
        )
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
    return {"median": float(sampled.median()), "p99": float(sampled.quantile(0.99))}


def scan_ms(scheme: Any, gallery: np.ndarray, query: np.ndarray) -> float:
    stored = scheme.encode(torch.from_numpy(gallery[:, :TIMED]).float())
    form = torch.from_numpy(query[:TIMED]).float().exp()
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        scheme.overlap(stored, form)
        timings.append((time.perf_counter() - start) * 1000)
    return round(min(timings), 3)


def reference_check(
    exact: np.ndarray, gallery: np.ndarray, device: torch.device
) -> dict[str, Any]:
    full = torch.from_numpy(exact).to(device).exp()
    half = torch.from_numpy(gallery[: len(exact)]).to(device).float().exp()
    differences = (
        torch.einsum("asv,bsv->sab", half, half).log()
        - torch.einsum("asv,bsv->sab", full, full).log()
    ).abs()
    return {
        "galaxies": len(full),
        "median": float(differences.median()),
        "max": float(differences.max()),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=24 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure(name: str, sample: int, fit: int, queries: int) -> dict[str, Any]:
    mode = next(mode for mode in predictions.MODES if mode.name == name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    galaxies, fitting = draw(sample, fit)
    started = time.perf_counter()
    gallery, exact = predict(galaxies, mode, check=CHECK)
    fitted, _ = predict(fitting, mode)
    seen = np.flatnonzero(pql.observed(mode.survey)[galaxies])
    rng = np.random.default_rng([SEED, predictions.MODES.index(mode)])
    picked = np.sort(rng.choice(seen, min(queries, len(seen)), replace=False))
    unseen, _ = predict(galaxies[picked], mode, hidden=True)
    prediction_s = time.perf_counter() - started

    chosen_slots = [
        selections(mode, int(galaxies[position]), rng) for position in picked
    ]
    masks = {}
    for size in chosen_slots[0]:
        mask = torch.zeros(len(picked), slot_count(mode), dtype=torch.bool)
        for row, selection in enumerate(chosen_slots):
            mask[row, torch.from_numpy(np.asarray(selection[size]))] = True
        masks[size] = mask.to(device)
    positions = torch.from_numpy(picked).to(device)
    redshift = torch.from_numpy(redshifts()[galaxies]).float().to(device)

    basis = compression.Basis(max(compression.RANKS))
    basis.fit(
        torch.from_numpy(fitted[start : start + GALAXY_CHUNK]).to(device)
        for start in range(0, len(fitting), GALAXY_CHUNK)
    )
    timed = {
        scheme.name: scheme
        for scheme in compression.schemes(mode, basis.to(torch.device("cpu")))
    }
    window = masks["16"] if kind(mode) in ("image", "spectrum") else None
    reference, reference_pairs, anywhere, anywhere_pairs, _, _ = log_overlaps(
        None, mode, gallery, picked, unseen, device, window
    )
    entries = {}
    for scheme in compression.schemes(mode, basis):
        found, pairs, crossed, crossed_pairs, bits, floored = log_overlaps(
            scheme, mode, gallery, picked, unseen, device, window
        )
        entries[scheme.name] = {
            "bytes_per_galaxy": bits / 8 / len(gallery),
            "scan_ms": scan_ms(timed[scheme.name], gallery, gallery[picked[0]]),
            "floored": floored,
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
                        mask,
                        positions,
                        redshift,
                    )
                    for side in found
                }
                for size, mask in masks.items()
            },
        }
        if crossed:
            entries[scheme.name]["sizes"]["16 anywhere"] = {
                side: summary(
                    crossed[side],
                    crossed_pairs[side],
                    anywhere["exact"],
                    anywhere_pairs["exact"],
                    window,
                    positions,
                    redshift,
                )
                for side in crossed
            }
    return {
        "galaxies": galaxy_count(),
        "sample": len(galaxies),
        "fit": len(fitting),
        "queries": len(picked),
        "prediction_s": round(prediction_s, 1),
        "fp16_reference_check": reference_check(exact, gallery, device)
        if kind(mode) == "image"
        else None,
        "schemes": entries,
        "environment": environment(),
        "device": str(device),
        "memory": memory(),
    }


def meets(entry: dict[str, Any]) -> bool:
    for result in entry["sizes"].values():
        both = result["both"]
        if (
            both["spearman_median"] < TARGETS["spearman_median"]
            or both["spearman_p5"] < TARGETS["spearman_p5"]
            or both["top10_median"] < TARGETS["top10_median"]
            or both["identity_exact"] - both["identity"] > TARGETS["identity"]
        ):
            return False
        if both["relevance_exact"] is not None and not (
            abs(both["relevance"] - both["relevance_exact"]) <= TARGETS["relevance"]
        ):
            return False
    return True


def worst(entry: dict[str, Any]) -> float:
    return min(result["both"]["spearman_median"] for result in entry["sizes"].values())


def choices(results: dict[str, dict[str, Any]], total: int) -> dict[str, Any]:
    accurate = {
        mode: min(
            (name for name, entry in entries.items() if meets(entry)),
            key=lambda name: entries[name]["bytes_per_galaxy"],
            default=None,
        )
        for mode, entries in results.items()
    }
    thresholds = sorted(
        {worst(entry) for entries in results.values() for entry in entries.values()},
        reverse=True,
    )
    budgets = {}
    for budget in BUDGETS:
        budgets[f"{budget / 1e9:g}_gb"] = None
        for threshold in thresholds:
            picks = {
                mode: min(
                    (
                        name
                        for name, entry in entries.items()
                        if worst(entry) >= threshold
                    ),
                    key=lambda name: entries[name]["bytes_per_galaxy"],
                    default=None,
                )
                for mode, entries in results.items()
            }
            if None in picks.values():
                continue
            size = total * sum(
                results[mode][name]["bytes_per_galaxy"] for mode, name in picks.items()
            )
            if size <= budget:
                budgets[f"{budget / 1e9:g}_gb"] = {
                    "worst_spearman_median": threshold,
                    "corpus_bytes": size,
                    "schemes": picks,
                }
                break
    return {"accuracy": accurate, "budgets": budgets}


def report(measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = next(iter(measured.values()))["galaxies"]
    return {
        "targets": TARGETS,
        "modes": measured,
        "choices": choices(
            {name: result["schemes"] for name, result in measured.items()}, total
        ),
    }


@app.local_entrypoint()
def main(sample: int = 1024, fit: int = 512, queries: int = 200) -> None:
    names = [mode.name for mode in predictions.MODES]
    measured = measure.map(
        names, kwargs={"sample": sample, "fit": fit, "queries": queries}
    )
    written = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **report(dict(zip(names, measured, strict=True))),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(written, indent=2) + "\n")
    print(f"wrote {REPORT}")
