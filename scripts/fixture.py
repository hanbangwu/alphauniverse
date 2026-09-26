import argparse
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL.Image import fromarray
from sklearn.preprocessing import normalize

from app.config import (
    ANCHOR,
    CROP_PIXELS,
    DIM,
    N_MORPHOLOGIES,
    N_PATCHES,
    SPECTRUM_SURVEYS,
    TOKEN_SURVEYS,
    artifact,
    build_dir,
    points,
    store_schema,
)
from app.cutouts import encode, write_cutouts
from app.search import generate_index, source, starts, with_spectrum
from app.spectra import write_spectra

TOKENS: dict[str, int] = {
    ANCHOR: N_PATCHES + 12,
    "hsc": N_PATCHES + 13,
    "desi": 273,
    "sdss": 273,
}

STRIDE: dict[str, int] = {ANCHOR: 1, "hsc": 2, "desi": 3, "sdss": 4}

CLUSTERS = 64
NOISE = 0.35
SOURCE_PIXELS = CROP_PIXELS + 32
PIXEL_NOISE = 4
SAMPLES = 512
MASKED = 4


def _frames(seed: int, galaxies: int) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    ramp = np.linspace(0, 255, SOURCE_PIXELS, dtype=np.float32)
    base = (ramp[:, None, None] + ramp[None, :, None]) / 2
    for _ in range(galaxies):
        offset = rng.integers(0, 256)
        noise = rng.integers(0, PIXEL_NOISE, (SOURCE_PIXELS, SOURCE_PIXELS, 3))
        yield ((base + offset + noise) % 256).astype(np.uint8)


def covered(survey: str, galaxy: int) -> bool:
    return galaxy % STRIDE[survey] == 0


def _spectra(seed: int, galaxies: int) -> dict[str, list[dict[str, np.ndarray] | None]]:
    rng = np.random.default_rng(seed + 2)
    wavelength = np.linspace(3600, 9800, SAMPLES, dtype=np.float32)
    cells: dict[str, list[dict[str, np.ndarray] | None]] = {
        survey: [] for survey in SPECTRUM_SURVEYS
    }
    for galaxy in range(galaxies):
        for survey in SPECTRUM_SURVEYS:
            if not covered(survey, galaxy):
                cells[survey].append(None)
                continue
            flux = rng.standard_normal(SAMPLES).astype(np.float32)
            flux[rng.choice(SAMPLES, MASKED, replace=False)] = np.nan
            cells[survey].append({"wavelength": wavelength, "flux": flux})
    return cells


def _cells(
    rng: np.random.Generator, centres: np.ndarray, galaxies: int
) -> tuple[dict[str, list[np.ndarray | None]], dict[str, list[np.ndarray | None]]]:
    embeddings: dict[str, list[np.ndarray | None]] = {
        survey: [] for survey in TOKEN_SURVEYS
    }
    tokens: dict[str, list[np.ndarray | None]] = {
        survey: [] for survey in TOKEN_SURVEYS
    }

    for galaxy in range(galaxies):
        for survey, count in TOKENS.items():
            if not covered(survey, galaxy):
                embeddings[survey].append(None)
                tokens[survey].append(None)
                continue
            assigned = rng.integers(len(centres), size=count)
            rows = centres[assigned] + NOISE * rng.standard_normal((count, DIM))
            embeddings[survey].append(rows.astype(np.float16))
            tokens[survey].append(assigned.astype(np.uint32))

    return embeddings, tokens


def _store(
    role: str,
    cells: dict[str, list[np.ndarray | None]],
    galaxies: int,
    flags: dict[str, np.ndarray],
) -> None:
    pq.write_table(
        pa.table(
            {
                "galaxy": np.arange(galaxies),
                **{
                    survey: [
                        None if cell is None else list(cell) for cell in cells[survey]
                    ]
                    for survey in TOKEN_SURVEYS
                },
                **flags,
            },
            schema=store_schema(role),
        ),
        artifact(role),
        compression="zstd",
    )


def _project(basis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return normalize(rows) @ basis


def build(galaxies: int, seed: int = 0) -> Path:
    target = build_dir()
    target.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    centres = normalize(rng.standard_normal((CLUSTERS, DIM)))

    embeddings, tokens = _cells(rng, centres, galaxies)

    rows = np.arange(galaxies)
    labelled = rows % 10 != 0
    flags = {"gz10": labelled, "provabgs": rows % 3 != 0}

    _store("encoded", embeddings, galaxies, flags)
    _store("tokens", tokens, galaxies, flags)

    write_cutouts([encode(fromarray(frame)) for frame in _frames(seed, galaxies)])
    write_spectra(_spectra(seed, galaxies))

    category = pa.array(rng.integers(N_MORPHOLOGIES, size=galaxies), mask=~labelled)

    anchors = np.stack(
        [np.asarray(cell, dtype=np.float32) for cell in embeddings[ANCHOR]]
    )
    basis = rng.standard_normal((DIM, 2)).astype(np.float32)

    pq.write_table(
        points(rows, _project(basis, anchors.mean(axis=1)), category),
        artifact("mean_points"),
        compression="zstd",
    )

    owner = np.repeat(rows, anchors.shape[1])
    pq.write_table(
        points(owner, _project(basis, anchors.reshape(-1, DIM)), category.take(owner)),
        artifact("full_points"),
        compression="zstd",
    )

    for cached in (source, with_spectrum, starts):
        cached.cache_clear()
    generate_index()
    for cached in (source, with_spectrum, starts):
        cached.cache_clear()
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Builds a small artifact tree in the schemas that app/ serves."
    )
    parser.add_argument("--galaxies", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(".cache") / "fixture")
    arguments = parser.parse_args()

    os.environ["ALPHAUNIVERSE_CACHE"] = str(arguments.out.resolve())
    target = build(arguments.galaxies, arguments.seed)

    total = sum(path.stat().st_size for path in target.iterdir())
    print(f"{arguments.galaxies} galaxies -> {target} ({total / 1e6:.1f} MB)")
    for path in sorted(target.iterdir()):
        print(f"  {path.name:24} {path.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
