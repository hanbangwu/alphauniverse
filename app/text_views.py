from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from io import BytesIO

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from astropy.convolution import Gaussian1DKernel, convolve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import pql, search, templates
from .config import (
    GEMMA_DIM,
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    MODE_SURVEYS,
    MODES,
    PLOT_INCHES,
    REDSHIFT,
    SEED,
    SPECTRUM_GRID,
    SPECTRUM_MODES,
    SPECTRUM_SURVEYS,
    TABLE_MODES,
    TABLE_VALUE_SURVEYS,
    TEXT_VIEWS,
    TOP_CODES,
    VALIDATION,
    VOCABULARY,
    SpectrumSurvey,
    artifact,
    device,
    galaxy_count,
)
from .dataset import dataset, image, predicted_image, samples
from .observations import table_values
from .text_search import text_model

KINDS = ("observed", "template", "sample")
COMBINATIONS = {
    "images": tuple(IMAGE_MODES),
    "spectra": tuple(SPECTRUM_MODES),
    "catalogues": tuple(TABLE_MODES),
    "image_and_spectrum": ("ls_image", "desi_spectrum"),
    "everything": MODES,
}
VIEW_TYPES = (
    *(f"{kind} {mode}" for mode in MODES for kind in KINDS),
    *(f"combination {name}" for name in COMBINATIONS),
)
SAMPLE_WEIGHTS = {mode: 8 if mode in TABLE_MODES else 4 for mode in MODES}
SMOOTHING = (1, 2, 3)
PRECISIONS = (3, 4, 5)
SPECTRUM_RANGES: dict[SpectrumSurvey, tuple[float, float]] = {
    "desi": (3600.0, 9824.0),
    "sdss": (3795.8, 9221.5),
}
FLUX_PERCENTILES = (0.5, 99.9)
FLUX_PADDING = 0.1
ROUND = 40_000
PART = 4_000
CHUNK = 400
PLOTTERS = 7
PROTOTYPE_HELD_OUT = np.concatenate(
    [
        [110, 301, 319, 830, 1003, 1365, 1413, 1492, 1561, 1653, 1655, 1656, 1790],
        [1843, 1918, 1988, 1989, 1991, 2006, 2026, 2027, 2092, 2173, 2219, 2247, 2414],
        [2481, 2725, 2726, 2863, 2918, 3095, 3166, 3221, 3286, 3333, 3923, 3955, 3962],
        [3965, 4093, 4102, 4459, 4670, 4891, 4926, 5026, 5091, 5179, 5238, 5245, 5287],
        [5371, 5467, 5643, 5670, 5678, 5710, 5932, 6313, 6319, 6694, 6705, 7134, 7139],
        [7184, 7190, 7493, 7520, 7650, 7666, 7810, 7896, 7938, 7983, 8081, 8087, 8123],
        [8226, 8384, 8414, 8494, 8517, 8680, 9305, 9452, 9679, 9701, 9938, 10004],
        [10097, 10104, 10469, 10474, 10602, 10716, 10721, 10745, 10942, 11218, 11581],
        [12137, 12230, 12576, 12628, 12641, 12660, 12689, 12846, 12916, 12970, 13105],
        [13190, 13455, 13471, 13540, 13545, 13592, 13989, 14103, 14113, 14268, 14308],
        [14381, 14383, 14631, 14680, 14733, 14908, 15169, 15304, 15307, 15598, 15614],
        [15768, 16021, 16075, 16236, 16304, 16464, 16928, 17239, 17243],
    ]
)


def held_out() -> np.ndarray:
    _, split = train_test_split(
        np.arange(galaxy_count()), test_size=VALIDATION, random_state=SEED
    )
    return np.union1d(split, PROTOTYPE_HELD_OUT)


def weights(galaxies: np.ndarray) -> np.ndarray:
    count = len(galaxies)
    columns = [
        column
        for mode in MODES
        for column in (
            pql.observed(MODE_SURVEYS[mode])[galaxies],
            np.ones(count),
            np.full(count, SAMPLE_WEIGHTS[mode]),
        )
    ]
    columns += [np.ones(count)] * len(COMBINATIONS)
    return np.stack(columns, axis=1).astype(np.float64)


def drawn(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cumulative = np.cumsum(probabilities, axis=-1, dtype=np.float64)
    cumulative /= cumulative[..., -1:]
    return (cumulative <= rng.random(probabilities.shape[:-1])[..., None]).sum(-1)


def rows(galaxies: np.ndarray) -> pa.RecordBatch:
    return pql.predictions().take(pa.array(galaxies)).combine_chunks().to_batches()[0]


def decoded(mode: str, codes: np.ndarray) -> np.ndarray:
    codec = templates.decoder(mode)
    vocabulary = IMAGE_VOCABULARY if mode in IMAGE_MODES else VOCABULARY
    codes = torch.from_numpy(codes.astype(np.int64)).to(device())
    latents = templates.codebook(codec, vocabulary)[codes].transpose(1, 2)
    if mode in IMAGE_MODES:
        return templates.decoded_images(codec, latents, IMAGE_MODES[mode])
    return templates.decoded_spectra(codec, latents)


def sampled_image_codes(
    found: pa.RecordBatch, mode: str, rng: np.random.Generator
) -> np.ndarray:
    codes, probabilities, spread = pql.query_image_tokens(
        found, IMAGE_MODES[mode], slice(None)
    )
    tail = spread * (IMAGE_VOCABULARY - TOP_CODES)
    chosen = drawn(np.concatenate((probabilities, tail[..., None]), axis=-1), rng)
    picked = np.take_along_axis(
        codes, np.minimum(chosen, TOP_CODES - 1)[..., None], axis=-1
    )[..., 0]
    rank = rng.integers(IMAGE_VOCABULARY - TOP_CODES, size=chosen.shape)
    shifted = np.sort(codes, axis=-1) - np.arange(TOP_CODES)
    beyond = rank + (shifted <= rank[..., None]).sum(axis=-1)
    return np.where(chosen == TOP_CODES, beyond, picked)


def sampled_spectrum_codes(
    found: pa.RecordBatch, mode: str, rng: np.random.Generator
) -> np.ndarray:
    survey = SPECTRUM_MODES[mode]
    mean, directions, _, _ = pql.basis()[survey]
    coefficients = pql.spectrum_token_coefficients(found, survey, slice(None))
    return drawn(np.clip(mean + coefficients @ directions, 0, None), rng)


def sampled_table_values(
    found: pa.RecordBatch, mode: str, rng: np.random.Generator
) -> np.ndarray:
    values = templates.bin_values(mode)
    probabilities = pql.table_value_probabilities(found, mode)
    chosen = drawn(np.where(np.isnan(values), 0, probabilities), rng)
    return values[np.arange(len(values)), chosen]


def augmented(picture: PIL.Image.Image, rng: np.random.Generator) -> PIL.Image.Image:
    choice = int(rng.integers(len(PIL.Image.Transpose) + 1))
    if choice == 0:
        return picture
    return picture.transpose(list(PIL.Image.Transpose)[choice - 1])


def plotted(
    wavelength: np.ndarray, flux: np.ndarray, survey: SpectrumSurvey, sigma: int
) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    low, high = SPECTRUM_RANGES[survey]
    shown = (wavelength >= low) & (wavelength <= high)
    flux = convolve(flux, Gaussian1DKernel(sigma), preserve_nan=True)[shown]
    bottom, top = np.nanpercentile(flux, FLUX_PERCENTILES)
    padding = FLUX_PADDING * (top - bottom)
    figure = Figure(figsize=PLOT_INCHES, layout="tight")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(wavelength[shown], flux, color="black", linewidth=1)
    axes.set(
        xlim=(low, high),
        ylim=(bottom - padding, top + padding),
        yticks=[],
        xlabel="Observed wavelength (Å)",
        ylabel=r"Flux density $f_\lambda$ (arbitrary units)",
    )
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def catalogue_text(mode: str, values: np.ndarray, rng: np.random.Generator) -> str:
    columns = (
        (REDSHIFT,) if mode == REDSHIFT else TABLE_VALUE_SURVEYS[MODE_SURVEYS[mode]]
    )
    precision = int(rng.choice(PRECISIONS))
    lines = [
        f"{column.partition('-')[0]}: {value:.{precision}g}"
        for column, value in zip(columns, values, strict=True)
        if np.isfinite(value)
    ]
    return "\n".join(rng.permutation(lines))


def realised(
    mode: str,
    kind: str,
    galaxies: np.ndarray,
    rng: np.random.Generator,
    pool: Executor,
) -> list[PIL.Image.Image] | list[str]:
    if mode in IMAGE_MODES:
        survey = IMAGE_MODES[mode]
        if kind == "observed":
            encoded = [image(int(galaxy), survey) for galaxy in galaxies]
        else:
            flux = (
                search.templates(mode)[galaxies]
                if kind == "template"
                else decoded(mode, sampled_image_codes(rows(galaxies), mode, rng))
            )
            encoded = [predicted_image(row, survey) for row in flux]
        return [augmented(PIL.Image.open(BytesIO(found)), rng) for found in encoded]
    if mode in SPECTRUM_MODES:
        survey = SPECTRUM_MODES[mode]
        if kind == "observed":
            cells = dataset().data.column(SPECTRUM_SURVEYS[survey])
            spectra = [samples(cells[int(galaxy)]).values() for galaxy in galaxies]
        else:
            flux = (
                search.templates(mode)[galaxies]
                if kind == "template"
                else decoded(mode, sampled_spectrum_codes(rows(galaxies), mode, rng))
            )
            spectra = [(SPECTRUM_GRID, row) for row in flux]
        wavelengths, fluxes = zip(*spectra)
        sigmas = rng.choice(SMOOTHING, len(galaxies)).tolist()
        plots = pool.map(plotted, wavelengths, fluxes, [survey] * len(galaxies), sigmas)
        return [PIL.Image.fromarray(found) for found in plots]
    if kind == "observed":
        order = np.argsort(galaxies)
        values = table_values(mode, galaxies[order])[0][np.argsort(order)]
    elif kind == "template":
        values = search.templates(mode)[galaxies]
    else:
        values = sampled_table_values(rows(galaxies), mode, rng)
    return [catalogue_text(mode, row, rng) for row in values]


def joined(documents: list[PIL.Image.Image | str]) -> dict | str:
    pictures = [found for found in documents if isinstance(found, PIL.Image.Image)]
    texts = [found for found in documents if isinstance(found, str)]
    text = "\n".join(["<|image|>"] * len(pictures) + texts)
    return {"text": text, "image": pictures} if pictures else text


def combined(
    members: tuple[str, ...],
    galaxies: np.ndarray,
    rng: np.random.Generator,
    pool: Executor,
) -> list[dict | str]:
    parts = []
    for mode in members:
        kinds = np.where(
            pql.observed(MODE_SURVEYS[mode])[galaxies],
            rng.integers(len(KINDS), size=len(galaxies)),
            rng.integers(1, len(KINDS), size=len(galaxies)),
        )
        documents = [None] * len(galaxies)
        for index, kind in enumerate(KINDS):
            chosen = np.flatnonzero(kinds == index)
            if len(chosen):
                for position, found in zip(
                    chosen, realised(mode, kind, galaxies[chosen], rng, pool)
                ):
                    documents[position] = found
        parts.append(documents)
    return [joined(list(documents)) for documents in zip(*parts)]


def documents(
    view_type: str, galaxies: np.ndarray, rng: np.random.Generator, pool: Executor
) -> list:
    kind, name = view_type.split(" ", 1)
    if kind == "combination":
        return combined(COMBINATIONS[name], galaxies, rng, pool)
    return realised(name, kind, galaxies, rng, pool)


def part(
    index: int, galaxies: np.ndarray, found: np.ndarray, pool: Executor
) -> pa.Table:
    rng = np.random.default_rng([SEED, index])
    chosen = rng.choice(found.size, size=PART, p=(found / found.sum()).ravel())
    owners, view_types = np.divmod(chosen, len(VIEW_TYPES))
    owners = galaxies[owners]
    embeddings = np.empty((PART, GEMMA_DIM), np.float16)
    for view_type in np.unique(view_types):
        members = np.flatnonzero(view_types == view_type)
        for start in range(0, len(members), CHUNK):
            batch = members[start : start + CHUNK]
            embeddings[batch] = text_model().encode(
                documents(VIEW_TYPES[view_type], owners[batch], rng, pool),
                prompt_name="Document",
            )
    return pa.table(
        {
            "galaxy": owners,
            "view_type": view_types,
            "round": np.full(PART, index // (ROUND // PART)),
            "embedding": pa.FixedSizeListArray.from_arrays(
                embeddings.ravel(), GEMMA_DIM
            ),
        },
        schema=TEXT_VIEWS,
    )


def generate_text_views(views: int) -> Iterator[None]:
    folder = artifact("text_views")
    folder.mkdir(exist_ok=True)
    galaxies = np.setdiff1d(np.arange(galaxy_count()), held_out())
    found = weights(galaxies)
    with ProcessPoolExecutor(PLOTTERS) as pool:
        for index in tqdm(range(views // PART), desc="text views"):
            path = folder / f"part-{index:05d}.parquet"
            if path.exists():
                continue
            hidden = folder / f".{path.name}"
            pq.write_table(part(index, galaxies, found, pool), hidden)
            hidden.rename(path)
            yield
