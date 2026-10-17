import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow.compute as pc
import torch
from tqdm import tqdm

from app import observations, templates
from app.config import (
    CROP_PIXELS,
    DATASET_REVISION,
    IMAGE_MODES,
    MODE_SURVEYS,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    REDSHIFT_COLUMNS,
    SPECTRUM_MODES,
    TABLE_MODES,
    TABLE_VALUE_SURVEYS,
    artifact,
    device,
)
from app.dataset import usable
from app.search import tokens
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import environment, git, memory

app = modal.App("alphauniverse-template-quality")
image = build_image.add_local_python_source("modal_app")

REPORT = Path("docs/benchmarks/template_quality.json")
BATCH = 256
RADII = np.arange(0, CROP_PIXELS // 2 + 4, 4)
REST = np.arange(2000, 10600, 100)
PERCENTILES = (5, 50, 95)
QUARTILES = 4
MODELS = ("template", "floor", "null")


def percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"p{share}": round(float(np.percentile(values, share)), 4)
        for share in PERCENTILES
    }


def chunks(galaxies: np.ndarray) -> list[np.ndarray]:
    return [galaxies[start : start + BATCH] for start in range(0, len(galaxies), BATCH)]


def own_codes(mode: str, galaxies: np.ndarray, first: int, count: int) -> np.ndarray:
    cells = observations.taken(tokens().column(MODE_SURVEYS[mode]), galaxies)
    codes = pc.list_slice(cells, first, first + count).flatten().to_numpy()
    return codes.reshape(len(galaxies), count).astype(np.int64)


class Profile:
    def __init__(self, edges: np.ndarray) -> None:
        self.edges = edges
        self.counts = np.zeros(len(edges) - 1)
        self.sums = {model: np.zeros(len(edges) - 1) for model in MODELS[:2]}

    def add(
        self, positions: np.ndarray, counts: np.ndarray, sums: dict[str, np.ndarray]
    ) -> None:
        self.counts += np.histogram(positions, self.edges, weights=counts)[0]
        for model, total in self.sums.items():
            total += np.histogram(positions, self.edges, weights=sums[model])[0]

    def report(self) -> dict[str, list[float]]:
        shown = self.counts > 0
        return {
            "centres": ((self.edges[:-1] + self.edges[1:]) / 2)[shown].tolist(),
            **{
                model: np.round(sums[shown] / self.counts[shown], 4).tolist()
                for model, sums in self.sums.items()
            },
        }


def summary(chi2: dict[str, np.ndarray]) -> dict[str, Any]:
    measured = np.all([np.isfinite(values) for values in chi2.values()], axis=0)
    chi2 = {model: values[measured] for model, values in chi2.items()}
    recovered = (chi2["null"] - chi2["template"]) / (chi2["null"] - chi2["floor"])
    return {
        "galaxies": int(measured.sum()),
        "chi2": {model: percentiles(values) for model, values in chi2.items()},
        "recovered": percentiles(recovered[np.isfinite(recovered)]),
    }


def by_snr(chi2: dict[str, np.ndarray], snr: np.ndarray) -> list[dict[str, Any]]:
    edges = np.quantile(snr[np.isfinite(snr)], np.linspace(0, 1, QUARTILES + 1))
    quartile = np.digitize(snr, edges[1:-1])
    return [
        {"snr": [round(float(edge), 3) for edge in edges[index : index + 2]]}
        | summary({model: values[quartile == index] for model, values in chi2.items()})
        for index in range(QUARTILES)
    ]


def image_quality(mode: str, survey: str) -> dict[str, Any]:
    codec = templates.decoder(mode)
    stored = np.load(artifact(f"template_{mode}"), mmap_mode="r")
    null = stored.mean(axis=0, dtype=np.float64)
    centre = (CROP_PIXELS - 1) / 2
    rows, columns = np.indices((CROP_PIXELS, CROP_PIXELS))
    radius = np.broadcast_to(
        np.hypot(rows - centre, columns - centre), stored.shape[1:]
    ).ravel()
    galaxies = observations.observed(mode)
    profile = Profile(RADII)
    chi2 = {model: np.empty(len(galaxies)) for model in MODELS}
    for start in tqdm(range(0, len(galaxies), BATCH), desc=mode):
        part = galaxies[start : start + BATCH]
        flux, ivar = observations.images(survey, part)
        codes = torch.from_numpy(own_codes(mode, part, 0, N_IMAGE_TOKENS))
        models = {
            "template": stored[part],
            "floor": templates.decoded_images(
                codec, codec.quantizer.decode(codes.to(device())), survey
            ),
            "null": null,
        }
        valid = ivar > 0
        weights = np.sqrt(np.where(valid, ivar, 0))
        residuals = {
            model: (flux - values) * weights for model, values in models.items()
        }
        for model, values in residuals.items():
            chi2[model][start : start + len(part)] = (values**2).sum(
                axis=(1, 2, 3)
            ) / valid.sum(axis=(1, 2, 3))
        profile.add(
            radius,
            valid.sum(axis=0).ravel(),
            {
                model: values.sum(axis=0, dtype=np.float64).ravel()
                for model, values in residuals.items()
            },
        )
    return summary(chi2) | {"radius": profile.report()}


def spectrum_quality(mode: str, survey: str) -> dict[str, Any]:
    codec = templates.decoder(mode)
    grid = codec.latent_grid.wavelength.cpu().numpy()
    stored = np.load(artifact(f"template_{mode}"), mmap_mode="r")
    null = stored.mean(axis=0, dtype=np.float64)
    value, _, warning = REDSHIFT_COLUMNS[survey]
    galaxies = observations.observed(mode)
    profile = Profile(REST)
    chi2 = {model: np.full(len(galaxies), np.nan) for model in MODELS}
    snr = np.full(len(galaxies), np.nan)
    for start in tqdm(range(0, len(galaxies), BATCH), desc=mode):
        part = galaxies[start : start + BATCH]
        spectra = observations.spectra(survey, part)
        codes = torch.from_numpy(own_codes(mode, part, 1, N_SPECTRUM_TOKENS))
        models = {
            "template": stored[part],
            "floor": templates.decoded_spectra(
                codec, codec.quantizer.decode(codes.to(device()))
            ),
            "null": np.broadcast_to(null, (len(part), len(null))),
        }
        for row, measured in enumerate(observations.redshift_rows(part)):
            wavelength, flux, ivar = observations.samples(spectra, row)
            if len(flux) < 2:
                continue
            snr[start + row] = np.median(flux * np.sqrt(ivar))
            residuals = {}
            for model, values in models.items():
                shape = np.interp(wavelength, grid, values[row])
                amplitude = (ivar * flux * shape).sum() / (ivar * shape**2).sum()
                residuals[model] = (flux - amplitude * shape) * np.sqrt(ivar)
                chi2[model][start + row] = (
                    residuals[model] @ residuals[model] / (len(flux) - 1)
                )
            if usable(measured[value], measured[warning]):
                profile.add(
                    wavelength / (1 + measured[value]), np.ones(len(flux)), residuals
                )
    return summary(chi2) | {
        "rest_wavelength": profile.report(),
        "by_snr": by_snr(chi2, snr),
    }


def table_quality(mode: str) -> dict[str, Any]:
    galaxies = observations.observed(mode)
    measured, errors = observations.table_values(mode, galaxies)
    values = templates.bin_values(mode)
    first = N_IMAGE_TOKENS if MODE_SURVEYS[mode] in IMAGE_MODES.values() else 0
    own = np.concatenate(
        [own_codes(mode, part, first, len(values)) for part in chunks(galaxies)]
    )
    stored = np.load(artifact(f"template_{mode}"), mmap_mode="r")
    models = {
        "template": stored[galaxies],
        "floor": values[np.arange(len(values)), own],
        "null": stored.mean(axis=0, dtype=np.float64),
    }
    pulls = {model: (measured - found) / errors for model, found in models.items()}
    pulls = {
        model: np.where(np.isfinite(pull), pull, np.nan)
        for model, pull in pulls.items()
    }
    typical = {
        model: np.nanmedian(np.abs(pull), axis=0) for model, pull in pulls.items()
    }
    names = TABLE_VALUE_SURVEYS.get(MODE_SURVEYS[mode], (mode,))
    return summary(
        {model: np.nanmean(pull**2, axis=1) for model, pull in pulls.items()}
    ) | {
        "median_absolute_pull": {
            name: {
                model: round(float(found[index]), 4) for model, found in typical.items()
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
        **{mode: table_quality(mode) for mode in TABLE_MODES},
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
