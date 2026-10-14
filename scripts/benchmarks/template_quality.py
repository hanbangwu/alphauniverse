import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from tqdm import tqdm

from app import pql, templates
from app.config import (
    BANDS,
    CROP_PIXELS,
    DATASET_REVISION,
    HSC,
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    N_IMAGE_TOKENS,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    REDSHIFT_LIMIT,
    SPECTRUM_MODES,
    TABLE_VALUE_SURVEYS,
    TOKEN_SURVEYS,
    VOCABULARY,
    artifact,
    device,
)
from app.dataset import dataset
from app.search import tokens
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-template-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/template_quality.json")
BATCH = 256
RADII = np.arange(0, CROP_PIXELS // 2 + 4, 4)
REST = np.arange(2000, 10600, 100)
QUANTILES = (0.05, 0.5, 0.95)
MODELS = ("template", "floor", "null")
MAGNITUDE_ERRORS = {
    f"{band}_cmodel_mag{HSC}": f"{band}_cmodel_magerr{HSC}" for band in "grizy"
}


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"p{round(100 * share)}": round(float(np.quantile(values, share)), 4)
        for share in QUANTILES
    }


def chunks(galaxies: np.ndarray) -> list[np.ndarray]:
    return [galaxies[start : start + BATCH] for start in range(0, len(galaxies), BATCH)]


def cells(column: str, galaxies: np.ndarray) -> pa.StructArray:
    return dataset().data.column(column).take(pa.array(galaxies)).combine_chunks()


def pixels(images: pa.StructArray, field: str, bands: int) -> np.ndarray:
    values = images.field(field).flatten().flatten().flatten().to_numpy()
    size = round((len(values) / (len(images) * bands)) ** 0.5)
    start = (size - CROP_PIXELS) // 2
    crop = slice(start, start + CROP_PIXELS)
    return values.reshape(len(images), bands, size, size)[:, :, crop, crop]


def image_data(survey: str, galaxies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    images = cells(TOKEN_SURVEYS[survey], galaxies)
    names = np.asarray(images.field("band").flatten()).reshape(len(images), -1)
    if not (np.char.upper(names.astype(str)) == BANDS[survey]).all():
        raise ValueError(f"{survey} bands are not stored as {BANDS[survey]}")
    return (
        pixels(images, "flux", len(BANDS[survey])),
        pixels(images, "ivar", len(BANDS[survey])),
    )


def own_tokens(survey: str, galaxies: np.ndarray) -> np.ndarray:
    found = tokens().column(survey).take(pa.array(galaxies)).combine_chunks()
    return found.flatten().to_numpy().reshape(len(galaxies), -1).astype(np.int64)


def observed_galaxies(mode: str) -> np.ndarray:
    if mode in pql.OBSERVED_IN:
        return np.flatnonzero(pql.observed(pql.OBSERVED_IN[mode]))
    return np.arange(pql.predictions().num_rows)


def column_values(column: str, galaxies: np.ndarray) -> np.ndarray:
    taken = dataset().data.column(column).take(pa.array(galaxies))
    return pc.fill_null(taken, np.nan).to_numpy().astype(np.float64)


def robust_scatter(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return 1.4826 * float(np.median(np.abs(finite - np.median(finite))))


def mean_template(mode: str) -> np.ndarray:
    stored = np.load(artifact(f"template_{mode}"), mmap_mode="r")
    total = np.zeros(stored.shape[1:])
    for part in chunks(np.arange(len(stored))):
        total += stored[part].sum(axis=0, dtype=np.float64)
    return (total / len(stored)).astype(np.float32)


class Profile:
    def __init__(self, edges: np.ndarray) -> None:
        self.edges = edges
        self.sums = {model: np.zeros((3, len(edges) - 1)) for model in MODELS}

    def add(self, model: str, positions: np.ndarray, residuals: np.ndarray) -> None:
        index = np.digitize(positions, self.edges) - 1
        kept = (index >= 0) & (index < len(self.edges) - 1)
        for row, weights in enumerate(
            (np.ones_like(residuals), residuals, residuals**2)
        ):
            self.sums[model][row] += np.bincount(
                index[kept], weights[kept], minlength=len(self.edges) - 1
            )

    def report(self) -> dict[str, Any]:
        counts = self.sums[MODELS[0]][0]
        shown = counts > 0
        return {
            "edges": self.edges.tolist(),
            "pixels": counts[shown].astype(int).tolist(),
            **{
                model: {
                    "mean": np.round(sums[1][shown] / sums[0][shown], 4).tolist(),
                    "rms": np.round(
                        np.sqrt(sums[2][shown] / sums[0][shown]), 4
                    ).tolist(),
                }
                for model, sums in self.sums.items()
            },
        }


def summary(chi2: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    found = {model: np.concatenate(values) for model, values in chi2.items()}
    recovered = (found["null"] - found["template"]) / (found["null"] - found["floor"])
    return {
        "galaxies": len(found["template"]),
        "chi2": {model: quantiles(values) for model, values in found.items()},
        "recovered": quantiles(recovered[np.isfinite(recovered)]),
    }


def image_quality(mode: str, survey: str) -> dict[str, Any]:
    codec = templates.decoder(mode)
    vectors = templates.codebook(codec, IMAGE_VOCABULARY)
    stored = np.load(artifact(f"template_{mode}"), mmap_mode="r")
    null = mean_template(mode)
    centre = (CROP_PIXELS - 1) / 2
    rows, columns = np.indices((CROP_PIXELS, CROP_PIXELS))
    radius = np.broadcast_to(
        np.hypot(rows - centre, columns - centre), (len(BANDS[survey]), *rows.shape)
    )
    profile = Profile(RADII)
    chi2 = {model: [] for model in MODELS}
    for part in tqdm(chunks(observed_galaxies(mode)), desc=mode):
        flux, ivar = image_data(survey, part)
        codes = torch.from_numpy(own_tokens(survey, part)[:, :N_IMAGE_TOKENS])
        models = {
            "template": stored[part],
            "floor": templates.decoded_images(
                codec, vectors[codes.to(device())].transpose(1, 2), survey
            ),
            "null": np.broadcast_to(null, flux.shape),
        }
        valid = ivar > 0
        for model, values in models.items():
            residuals = (flux - values) * np.sqrt(ivar)
            chi2[model].append(
                np.where(valid, residuals**2, 0).sum(axis=(1, 2, 3))
                / valid.sum(axis=(1, 2, 3))
            )
            profile.add(
                model, np.broadcast_to(radius, residuals.shape)[valid], residuals[valid]
            )
    return summary(chi2) | {"radius": profile.report()}


def spectrum_quality(mode: str, survey: str) -> dict[str, Any]:
    codec = templates.decoder(mode)
    vectors = templates.codebook(codec, VOCABULARY)
    grid = codec.latent_grid.wavelength.cpu().numpy()
    stored = np.load(artifact(f"template_{mode}"), mmap_mode="r")
    null = mean_template(mode)
    value, _, warning = REDSHIFT_COLUMNS[survey]
    profile = Profile(REST)
    chi2 = {model: [] for model in MODELS}
    for part in tqdm(chunks(observed_galaxies(mode)), desc=mode):
        spectra = cells(TOKEN_SURVEYS[survey], part)
        codes = torch.from_numpy(own_tokens(survey, part)[:, 1:]).to(device())
        models = {
            "template": stored[part],
            "floor": templates.decoded_spectra(codec, vectors[codes].transpose(1, 2)),
            "null": np.broadcast_to(null, (len(part), len(null))),
        }
        redshifts = column_values(value, part)
        flagged = pc.fill_null(
            dataset().data.column(warning).take(pa.array(part)), True
        ).to_numpy(zero_copy_only=False)
        rest = np.isfinite(redshifts) & ~flagged & (redshifts <= REDSHIFT_LIMIT)
        found = {model: np.empty(len(part)) for model in MODELS}
        for row in range(len(part)):
            wavelength, flux, ivar, mask = (
                np.asarray(spectra.field(field)[row].values)
                for field in ("lambda", "flux", "ivar", "mask")
            )
            valid = (wavelength > 0) & (ivar > 0) & ~mask
            for model, values in models.items():
                shape = np.interp(wavelength[valid], grid, values[row])
                amplitude = (ivar[valid] * flux[valid] * shape).sum() / (
                    ivar[valid] * shape**2
                ).sum()
                residuals = (flux[valid] - amplitude * shape) * np.sqrt(ivar[valid])
                found[model][row] = residuals @ residuals / (valid.sum() - 1)
                if rest[row]:
                    profile.add(
                        model, wavelength[valid] / (1 + redshifts[row]), residuals
                    )
        for model in MODELS:
            chi2[model].append(found[model])
    return summary(chi2) | {"rest_wavelength": profile.report()}


def table_data(mode: str, galaxies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mode == REDSHIFT:
        measured = {}
        for survey, (value, error, warning) in REDSHIFT_COLUMNS.items():
            redshifts = column_values(value, galaxies)
            flagged = pc.fill_null(
                dataset().data.column(warning).take(pa.array(galaxies)), True
            ).to_numpy(zero_copy_only=False)
            usable = np.isfinite(redshifts) & ~flagged & (redshifts <= REDSHIFT_LIMIT)
            measured[survey] = (usable, redshifts, column_values(error, galaxies))
        usable, redshifts, errors = measured["desi"]
        _, others, other_errors = measured["sdss"]
        return (
            np.where(usable, redshifts, others)[:, None],
            np.where(usable, errors, other_errors)[:, None],
        )
    columns = TABLE_VALUE_SURVEYS[mode.removesuffix("_table")]
    values = np.stack([column_values(column, galaxies) for column in columns], axis=1)
    errors = [
        column_values(MAGNITUDE_ERRORS[column], galaxies)
        if column in MAGNITUDE_ERRORS
        else np.full(len(galaxies), robust_scatter(values[:, index]))
        for index, column in enumerate(columns)
    ]
    return values, np.stack(errors, axis=1)


def table_quality(mode: str) -> dict[str, Any]:
    galaxies = observed_galaxies(mode)
    measured, errors = table_data(mode, galaxies)
    values = templates.bin_values(mode)
    survey, first = (
        (REDSHIFT, 0)
        if mode == REDSHIFT
        else (mode.removesuffix("_table"), N_IMAGE_TOKENS)
    )
    own = own_tokens(survey, galaxies)[:, first : first + len(values)]
    models = {
        "template": np.load(artifact(f"template_{mode}"), mmap_mode="r")[galaxies],
        "floor": values[np.arange(len(values)), own],
        "null": mean_template(mode),
    }
    pulls = {model: (measured - found) / errors for model, found in models.items()}
    chi2 = {
        model: [np.nanmean(np.where(np.isfinite(pull), pull**2, np.nan), axis=1)]
        for model, pull in pulls.items()
    }
    names = (
        [REDSHIFT]
        if mode == REDSHIFT
        else TABLE_VALUE_SURVEYS[mode.removesuffix("_table")]
    )
    return summary(chi2) | {
        "median_absolute_pull": {
            name: {
                model: round(
                    float(
                        np.nanmedian(
                            np.abs(
                                np.where(
                                    np.isfinite(pull[:, index]), pull[:, index], np.nan
                                )
                            )
                        )
                    ),
                    4,
                )
                for model, pull in pulls.items()
            }
            for index, name in enumerate(names)
        }
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=(16 * 1024, 32 * 1024),
    timeout=20 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def measure() -> dict[str, Any]:
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    modes = {
        **{mode: image_quality(mode, survey) for mode, survey in IMAGE_MODES.items()},
        **{
            mode: spectrum_quality(mode, survey)
            for mode, survey in SPECTRUM_MODES.items()
        },
        **{mode: table_quality(mode) for mode in templates.TABLE_MODALITIES},
    }
    return {
        "environment": environment(),
        "device": torch.cuda.get_device_name(),
        "modes": modes,
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
