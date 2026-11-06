import json
import math
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow as pa
import torch
import torchvision.transforms.functional as F

from app import encode, pql, predictions
from app.config import (
    ANCHOR,
    BANDS,
    CROP_PIXELS,
    DATASET_REVISION,
    GRID,
    N_IMAGE_TOKENS,
    TABLE_VALUE_SURVEYS,
    TOKEN_SURVEYS,
)
from app.dataset import dataset
from app.poses import QUARTERS, image_poses, spectrum_poses
from app.search import tokens
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import (
    environment,
    git,
    memory,
    observed_spectrum_tokens,
    redshifts,
    spectrum_query,
    wavelength,
    with_spectrum,
)

app = modal.App("alphauniverse-shape-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/shape_quality.json")
MODE = "ls_image"
CENTRE = np.array(
    [row * GRID + column for row in range(10, 14) for column in range(10, 14)]
)
TURNS = {
    "identity": (0, False),
    "quarter turn": (90, False),
    "half turn": (180, False),
    "three quarter turn": (270, False),
    "mirror": (0, True),
    "mirror and quarter turn": (90, True),
    "30 degrees": (30, False),
    "45 degrees": (45, False),
}
RADIUS, FIRST_SHAPE, SECOND_SHAPE = (
    next(column for column in TABLE_VALUE_SURVEYS[ANCHOR] if column.startswith(name))
    for name in ("SHAPE_R", "SHAPE_E1", "SHAPE_E2")
)
RADIUS_QUANTILE = 0.75
ELLIPTICITY = 0.2
WINDOWS = 16
MATCHES = 32
SAME_REDSHIFT = 0.01


def turned(images: torch.Tensor, angle: int, mirror: bool) -> torch.Tensor:
    images = images.flip(-1) if mirror else images
    if angle % 90 == 0:
        return torch.rot90(images, angle // 90, dims=(-2, -1))
    return F.rotate(images, angle, interpolation=F.InterpolationMode.BILINEAR)


def sources(size: int, angle: int, mirror: bool) -> np.ndarray:
    axis = torch.arange(size, dtype=torch.float32)
    pixels = torch.stack(torch.meshgrid(axis, axis, indexing="ij"))
    start = F.center_crop(pixels, [CROP_PIXELS, CROP_PIXELS])[:, 0, 0]
    moved = F.center_crop(turned(pixels, angle, mirror), [CROP_PIXELS, CROP_PIXELS])
    cells = moved.view(2, GRID, QUARTERS, GRID, QUARTERS).mean(dim=(2, 4))
    return (
        (cells - start[:, None, None]).permute(1, 2, 0).reshape(-1, 2).numpy() - 1.5
    ) / 4


def copied(galaxy: int, raw: dict, angle: int, mirror: bool) -> pa.RecordBatch:
    flux = torch.as_tensor(np.asarray(raw["flux"], np.float32))
    moved = raw | {"flux": list(turned(flux, angle, mirror).numpy())}
    codes = encode.image(encode.LegacySurveyImage, moved, BANDS[ANCHOR])[0].tolist()
    row = tokens().slice(galaxy, 1).to_pylist()[0]
    row[ANCHOR] = codes + row[ANCHOR][N_IMAGE_TOKENS:]
    predicted = predictions.predictions(predictions.inputs(row), predictions.TARGETS)
    fitted = pql.prediction_basis()
    return pql.prediction_batch([predictions.record(galaxy, predicted, fitted)])


def ranks(scores: np.ndarray, galaxy: int) -> int:
    return int((scores > scores[galaxy]).sum()) + 1


def searched(form: np.ndarray) -> dict[str, np.ndarray]:
    batches = pql.predictions().to_batches()
    found = pql.threaded(
        partial(
            pql.best_poses,
            mode=MODE,
            form=form,
            reads=pql.pose_reads(MODE, tuple(CENTRE.tolist())),
        ),
        batches,
    )
    return {
        "same shape": found[:, 0],
        "poses": found[:, 1].astype(np.int64),
        "same place": pql.threaded(
            lambda rows: pql.log_overlaps(rows, MODE, CENTRE, form).sum(axis=1), batches
        ),
        "any position": pql.threaded(
            lambda rows: pql.best_log_overlaps(rows, MODE, form).sum(axis=1), batches
        ),
    }


def oriented() -> np.ndarray:
    columns = dataset().data
    radius = columns.column(RADIUS).to_numpy(zero_copy_only=False)
    ellipticity = np.hypot(
        columns.column(FIRST_SHAPE).to_numpy(zero_copy_only=False),
        columns.column(SECOND_SHAPE).to_numpy(zero_copy_only=False),
    )
    large = radius >= np.nanquantile(radius, RADIUS_QUANTILE)
    return np.flatnonzero(large & (ellipticity >= ELLIPTICITY))


def agreement(form: np.ndarray, rows: pa.RecordBatch, cells: np.ndarray) -> float:
    return float(pql.log_overlaps(rows, MODE, cells, form).mean())


def image_quality(count: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    chosen = rng.choice(oriented(), count, replace=False)
    poses = image_poses(CENTRE).quarters / QUARTERS
    every = np.arange(N_IMAGE_TOKENS)
    found = {name: {"ranks": {}, "pose_error": [], "content": []} for name in TURNS}
    size = None
    for galaxy in chosen.tolist():
        raw = dataset().data.column(TOKEN_SURVEYS[ANCHOR])[galaxy].as_py()
        size = len(raw["flux"][0])
        own = pql.row(galaxy)
        other = pql.row(
            int(rng.choice(np.delete(np.arange(pql.predictions().num_rows), galaxy)))
        )
        forms = {}
        for name, (angle, mirror) in TURNS.items():
            if angle % 90 and size < math.ceil(CROP_PIXELS * math.sqrt(2)):
                continue
            row = copied(galaxy, raw, angle, mirror)
            scores = searched(pql.query_forms(row, {MODE: CENTRE})[MODE])
            for scorer in ("same shape", "same place", "any position"):
                found[name]["ranks"].setdefault(scorer, []).append(
                    ranks(scores[scorer], galaxy)
                )
            expected = sources(size, angle, mirror)
            error = np.linalg.norm(
                poses[scores["poses"][galaxy]] - expected[CENTRE], axis=-1
            )
            found[name]["pose_error"].append(float(error.mean()))
            if angle % 90 == 0:
                forms[name] = pql.query_forms(row, {MODE: every})[MODE]
                mapped = np.rint(expected).astype(int)
                found[name]["content"].append(
                    agreement(forms[name], own, mapped[:, 0] * GRID + mapped[:, 1])
                )
        for name in forms:
            reference = agreement(forms["identity"], own, every)
            unrelated = agreement(forms["identity"], other, every)
            found[name]["content"][-1] = (found[name]["content"][-1] - unrelated) / (
                reference - unrelated
            )
    return {
        "galaxies": len(chosen),
        "cutout_pixels": size,
        "radius_quantile": RADIUS_QUANTILE,
        "minimum_ellipticity": ELLIPTICITY,
        "transforms": {
            name: {
                "ranks": {
                    scorer: {
                        "first": round(float(np.mean(np.array(values) == 1)), 4),
                        "top_10": round(float(np.mean(np.array(values) <= 10)), 4),
                        "median": float(np.median(values)),
                    }
                    for scorer, values in parts["ranks"].items()
                },
                "median_pose_error_cells": round(
                    float(np.median(parts["pose_error"])), 3
                ),
                "content_share": round(float(np.median(parts["content"])), 4)
                if parts["content"]
                else None,
            }
            for name, parts in found.items()
            if parts["ranks"]
        },
    }


def spectrum_quality(count: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    measured = redshifts()
    queries = rng.choice(
        np.flatnonzero(with_spectrum() & np.isfinite(measured)), count, replace=False
    )
    stretch_offsets, shape_ratios, place_ratios, measured_shares = [], [], [], []
    for galaxy in queries.tolist():
        windows = observed_spectrum_tokens(wavelength(galaxy))
        start = int(rng.integers(len(windows) - WINDOWS + 1))
        slots = tuple(windows[start : start + WINDOWS].tolist())
        query = spectrum_query(galaxy, slots)
        mode, selected = next(iter(pql.selection(query).items()))
        found = pql.shape_sum(galaxy, mode, slots)
        top = pql.ranked(found.totals, galaxy, MATCHES)[1:]
        place = pql.ranked(
            pql.column_sums(galaxy, {mode: selected})[mode], galaxy, MATCHES
        )[1:]
        stretches = spectrum_poses(np.array(slots)).stretches[:, 0]
        known = np.isfinite(measured[top])
        measured_shares.append(known.mean())
        ratio = np.log1p(measured[top][known]) - np.log1p(measured[galaxy])
        stretch_offsets.append(
            np.abs(np.log(stretches[found.poses[top][known]]) - ratio)
        )
        shape_ratios.append(np.abs(ratio))
        same = place[np.isfinite(measured[place])]
        place_ratios.append(
            np.abs(np.log1p(measured[same]) - np.log1p(measured[galaxy]))
        )
    stretch, shape, aligned = (
        np.concatenate(parts) for parts in (stretch_offsets, shape_ratios, place_ratios)
    )
    return {
        "queries": len(queries),
        "windows": WINDOWS,
        "matches": MATCHES,
        "same_redshift": SAME_REDSHIFT,
        "measured_share": round(float(np.mean(measured_shares)), 4),
        "same_shape": {
            "stretch_matches_redshift": round(
                float(np.mean(stretch <= SAME_REDSHIFT)), 4
            ),
            "median_stretch_offset": round(float(np.median(stretch)), 5),
            "other_redshift": round(float(np.mean(shape > SAME_REDSHIFT)), 4),
        },
        "same_place": {
            "same_redshift": round(float(np.mean(aligned <= SAME_REDSHIFT)), 4)
        },
    }


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=(16 * 1024, 32 * 1024),
    timeout=2 * 60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure(galaxies: int, queries: int) -> dict[str, Any]:
    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "images": image_quality(galaxies),
        "spectra": spectrum_quality(queries),
        "memory": memory(),
    }


@app.local_entrypoint()
def main(galaxies: int = 32, queries: int = 100) -> None:
    report = {
        "commit": git("describe", "--always", "--dirty"),
        "revision": DATASET_REVISION,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **measure.remote(galaxies, queries),
    }
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {REPORT}")
