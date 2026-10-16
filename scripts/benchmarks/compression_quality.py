import json
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import modal
import numpy as np
import torch

from app import pql, predictions
from app.config import (
    DATASET_REVISION,
    PROBABILITY_FLOOR,
    SEED,
    STORE_COLUMNS,
    galaxy_count,
)
from app.dataset import spectrum
from app.search import tokens
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks import compression
from scripts.benchmarks.common import (
    environment,
    git,
    memory,
    observed_spectrum_tokens,
    redshifts,
)

app = modal.App("alphauniverse-compression-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/compression_quality.json")
TOP = 10
WINDOW = 16
SLOT_CHUNK = 32

Mode = predictions.Mode
kind = compression.kind


def slot_count(mode: Mode) -> int:
    return len(mode.keys) * len(mode.positions)


def draw(sample: int) -> np.ndarray:
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
    return np.sort(chosen)


def predict(galaxies: np.ndarray, mode: Mode, hidden: bool = False) -> np.ndarray:
    rows = tokens().select(STORE_COLUMNS).take(galaxies).to_pylist()
    found = np.empty(
        (len(galaxies), slot_count(mode), mode.vocabulary),
        dtype=np.float16 if kind(mode) == "image" else np.float32,
    )
    targets = {key: mode.positions for key in mode.keys}
    for position, row in enumerate(rows):
        context = predictions.inputs(row)
        if hidden:
            for key in mode.keys:
                context.pop(key, None)
        found[position] = predictions.distributions(
            predictions.predictions(context, targets), mode
        )
    return found


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


def exponentiated(log_probabilities: torch.Tensor) -> torch.Tensor:
    return log_probabilities.float().exp()


def shifted(log_probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    largest = log_probabilities.float().amax(dim=-1, keepdim=True)
    return (log_probabilities.float() - largest).double().exp(), largest[..., 0].T


def log_products(
    form: torch.Tensor, decoded: torch.Tensor, unseen: torch.Tensor, minimum: float
) -> tuple[torch.Tensor, torch.Tensor]:
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
    )


def products(
    form: torch.Tensor, decoded: torch.Tensor, unseen: torch.Tensor, minimum: float
) -> tuple[torch.Tensor, torch.Tensor]:
    overlaps = torch.einsum("qsv,gsv->sqg", form, decoded)
    own = torch.einsum("qsv,qsv->sq", form, unseen)
    return overlaps.clamp_min(minimum).log(), own.clamp_min(minimum).log()


class Overlaps(NamedTuple):
    found: torch.Tensor
    pairs: torch.Tensor
    crossed: torch.Tensor | None
    crossed_pairs: torch.Tensor | None
    bits: int


def log_overlaps(
    scheme: compression.Current | None,
    mode: Mode,
    gallery: np.ndarray,
    queries: np.ndarray,
    hidden: np.ndarray,
    device: torch.device,
    window: torch.Tensor | None = None,
) -> Overlaps:
    shape = (gallery.shape[1], len(queries))
    found = torch.empty(*shape, len(gallery), device=device)
    pairs = torch.empty(*shape, device=device)
    crossed = crossed_pairs = None
    if scheme is None and kind(mode) not in ("image", "spectrum"):
        overlap, minimum = log_products, torch.finfo(torch.float64).tiny
        exact = torch.Tensor.float
    else:
        overlap = products
        minimum = scheme.floor if scheme else PROBABILITY_FLOOR
        exact = exponentiated
    decode = scheme and scheme.decode
    query_decode = scheme and partial(scheme.decode, side="query")
    if window is not None:
        selected = [row.nonzero()[:, 0] for row in window]
        targets = anywhere_forms(scheme, gallery, queries, selected, device)
        crossed = torch.zeros(*shape, len(gallery), device=device)
        crossed_pairs = torch.zeros(*shape, device=device)
        for position, slots in enumerate(selected):
            crossed[slots, position] = -torch.inf
            crossed_pairs[slots, position] = -torch.inf
    bits = 0
    for part, rows, unseen in chunks(gallery, hidden, device):
        if scheme is None:
            decoded, decoded_unseen = exact(rows), exact(unseen)
            form = exact(rows[queries])
        else:
            stored = scheme.encode(rows)
            bits += scheme.bits(stored)
            decoded = decode(stored)
            decoded_unseen = decode(scheme.encode(unseen))
            form = query_decode(scheme.encode(rows[queries]))
        found[part], pairs[part] = overlap(form, decoded, decoded_unseen, minimum)
        if crossed is None:
            continue
        for position, (slots, target) in enumerate(zip(selected, targets, strict=True)):
            overlaps = torch.einsum("kv,gcv->kgc", target, decoded)
            crossed[slots, position] = torch.maximum(
                crossed[slots, position],
                overlaps.clamp_min(minimum).log().amax(dim=-1),
            )
            own = torch.einsum("kv,cv->kc", target, decoded_unseen[position])
            crossed_pairs[slots, position] = torch.maximum(
                crossed_pairs[slots, position],
                own.clamp_min(minimum).log().amax(dim=-1),
            )
    return Overlaps(found, pairs, crossed, crossed_pairs, bits)


def anywhere_forms(
    scheme: compression.Current | None,
    gallery: np.ndarray,
    queries: np.ndarray,
    selected: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    rows = [
        torch.from_numpy(gallery[query][slots.cpu().numpy()]).to(device)[None]
        for query, slots in zip(queries, selected, strict=True)
    ]
    if scheme is None:
        return [exponentiated(own)[0] for own in rows]
    return [scheme.decode(scheme.encode(own), side="query")[0] for own in rows]


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
    top = ranked.topk(min(TOP, ranked.shape[-1]), dim=-1).indices
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
    shared = (chosen(scores, TOP) & chosen(exact, TOP)).sum(dim=-1)
    result = {
        "spearman_median": float(correlation.median()),
        "spearman_p5": float(correlation.quantile(0.05)),
        f"top{TOP}_median": float((shared / min(TOP, galaxies - 1)).median()),
        "identity": float((unseen > scores.amax(dim=-1)).float().mean()),
        "identity_exact": float((unseen_exact > exact.amax(dim=-1)).float().mean()),
        "relevance": None,
        "relevance_exact": None,
    }
    own = redshift[positions]
    known = torch.isfinite(own)
    if known.any():
        result["relevance"] = float(
            relevance(scores, others, redshift, own)[known].nanmedian()
        )
        result["relevance_exact"] = float(
            relevance(exact, others, redshift, own)[known].nanmedian()
        )
    return result


@app.function(
    image=image,
    gpu="L4",
    cpu=16,
    memory=(32 * 1024, 128 * 1024),
    timeout=24 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure(name: str, sample: int, queries: int) -> dict[str, Any]:
    mode = next(mode for mode in predictions.MODES if mode.name == name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    galaxies = draw(sample)
    gallery = predict(galaxies, mode)
    seen = np.flatnonzero(pql.observed(mode.survey)[galaxies])
    rng = np.random.default_rng([SEED, predictions.MODES.index(mode)])
    picked = np.sort(rng.choice(seen, min(queries, len(seen)), replace=False))
    unseen = predict(galaxies[picked], mode, hidden=True)

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

    basis = (
        [torch.from_numpy(array).to(device) for array in pql.basis()[mode.survey][:2]]
        if kind(mode) == "spectrum"
        else []
    )
    scheme = compression.Current(mode, *basis)
    window = masks["16"] if kind(mode) in ("image", "spectrum") else None
    reference = log_overlaps(None, mode, gallery, picked, unseen, device, window)
    stored = log_overlaps(scheme, mode, gallery, picked, unseen, device, window)
    sizes = {
        size: summary(
            stored.found,
            stored.pairs,
            reference.found,
            reference.pairs,
            mask,
            positions,
            redshift,
        )
        for size, mask in masks.items()
    }
    if window is not None:
        sizes["16 anywhere"] = summary(
            stored.crossed,
            stored.crossed_pairs,
            reference.crossed,
            reference.crossed_pairs,
            window,
            positions,
            redshift,
        )
    return {
        "galaxies": galaxy_count(),
        "sample": len(galaxies),
        "queries": len(picked),
        "bytes_per_galaxy": stored.bits / 8 / len(gallery),
        "sizes": sizes,
        "environment": environment(),
        "device": str(device),
        "memory": memory(),
    }


@app.local_entrypoint()
def main(sample: int = 1024, queries: int = 200, modes: str = "") -> None:
    every = [mode.name for mode in predictions.MODES]
    names = list(dict.fromkeys(modes.split(","))) if modes else every
    if unknown := sorted(set(names) - set(every)):
        raise SystemExit(f"Unknown modes {unknown}; choose from {every}.")
    measured = measure.map(names, kwargs={"sample": sample, "queries": queries})
    written = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "modes": dict(zip(names, measured, strict=True)),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(written, indent=2) + "\n")
    print(f"wrote {REPORT}")
