import multiprocessing
from concurrent.futures import Executor, ProcessPoolExecutor
from functools import cache
from io import BytesIO

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.convolution import Gaussian1DKernel, convolve
from tqdm import tqdm

from . import alignment, pql, search, templates
from .config import (
    GEMMA_DIM,
    IMAGE_MODES,
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
    VIEW_COMBINATIONS,
    VIEW_KINDS,
    VIEW_TYPES,
    SpectrumSurvey,
    artifact,
)
from .dataset import dataset, image, predicted_image, samples
from .observations import table_values
from .text_search import text_model

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
    split = alignment.held_out(np.arange(len(dataset()))).cpu().numpy()
    return np.union1d(np.flatnonzero(split), PROTOTYPE_HELD_OUT)


def weights(galaxies: np.ndarray) -> np.ndarray:
    columns = []
    for kind, name in VIEW_TYPES:
        if kind == "observed":
            columns.append(pql.observed(MODE_SURVEYS[name])[galaxies])
        elif kind == "sample":
            columns.append(np.full(len(galaxies), SAMPLE_WEIGHTS[name]))
        else:
            columns.append(np.ones(len(galaxies)))
    return np.stack(columns, axis=1)


def rows(galaxies: np.ndarray, mode: str) -> pa.RecordBatch:
    table = pql.predictions()
    table = table.select(
        [TABLE_MODES[mode][0]]
        if mode in TABLE_MODES
        else [
            name
            for name in table.column_names
            if name.startswith(f"{MODE_SURVEYS[mode]}_")
        ]
    )
    sliced = [table.slice(galaxy, 1) for galaxy in galaxies.tolist()]
    return pa.concat_tables(sliced).combine_chunks().to_batches()[0]


def augmented(picture: PIL.Image.Image, rng: np.random.Generator) -> PIL.Image.Image:
    choice = rng.integers(len(PIL.Image.Transpose) + 1)
    return (
        picture if choice == 0 else picture.transpose(PIL.Image.Transpose(choice - 1))
    )


@cache
def spectrum_canvas(survey: SpectrumSurvey) -> tuple:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    canvas = FigureCanvasAgg(Figure(figsize=PLOT_INCHES, layout="tight"))
    axes = canvas.figure.subplots()
    (line,) = axes.plot([], [], color="black", linewidth=1)
    axes.set(
        xlim=SPECTRUM_RANGES[survey],
        yticks=[],
        xlabel="Observed wavelength (Å)",
        ylabel=r"Flux density $f_\lambda$ (arbitrary units)",
    )
    return canvas, axes, line


def plotted(
    wavelength: np.ndarray, flux: np.ndarray, survey: SpectrumSurvey, sigma: int
) -> PIL.Image.Image:
    low, high = SPECTRUM_RANGES[survey]
    shown = (wavelength >= low) & (wavelength <= high)
    flux = convolve(
        flux, Gaussian1DKernel(sigma), boundary="extend", preserve_nan=True
    )[shown]
    finite = flux[np.isfinite(flux)]
    bottom, top = np.percentile(finite, FLUX_PERCENTILES) if finite.size else (0, 1)
    padding = FLUX_PADDING * (top - bottom)
    canvas, axes, line = spectrum_canvas(survey)
    line.set_data(wavelength[shown], flux)
    axes.set_ylim(bottom - padding, top + padding)
    canvas.draw()
    return PIL.Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def catalogue_text(mode: str, values: np.ndarray, rng: np.random.Generator) -> str:
    columns = (
        (REDSHIFT,) if mode == REDSHIFT else TABLE_VALUE_SURVEYS[MODE_SURVEYS[mode]]
    )
    precision = rng.choice(PRECISIONS)
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
    if kind == "template":
        predicted = search.templates(mode)[galaxies]
    elif kind == "sample":
        predicted = templates.sampled(rows(galaxies, mode), mode, rng)
    if mode in IMAGE_MODES:
        survey = IMAGE_MODES[mode]
        encoded = (
            [image(galaxy, survey) for galaxy in galaxies.tolist()]
            if kind == "observed"
            else [predicted_image(flux, survey) for flux in predicted]
        )
        return [augmented(PIL.Image.open(BytesIO(found)), rng) for found in encoded]
    if mode in SPECTRUM_MODES:
        survey = SPECTRUM_MODES[mode]
        if kind == "observed":
            cells = dataset().data.column(SPECTRUM_SURVEYS[survey])
            wavelengths, fluxes = zip(
                *(samples(cells[galaxy]).values() for galaxy in galaxies.tolist())
            )
        else:
            wavelengths, fluxes = [SPECTRUM_GRID] * len(galaxies), predicted
        sigmas = rng.choice(SMOOTHING, len(galaxies))
        surveys = [survey] * len(galaxies)
        return list(pool.map(plotted, wavelengths, fluxes, surveys, sigmas))
    values = table_values(mode, galaxies)[0] if kind == "observed" else predicted
    return [catalogue_text(mode, row, rng) for row in values]


def joined(documents: tuple[PIL.Image.Image | str, ...]) -> dict | str:
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
        kinds = rng.integers(
            ~pql.observed(MODE_SURVEYS[mode])[galaxies], len(VIEW_KINDS)
        )
        member = [None] * len(galaxies)
        for kind in np.unique(kinds):
            chosen = np.flatnonzero(kinds == kind)
            found = realised(mode, VIEW_KINDS[kind], galaxies[chosen], rng, pool)
            for position, document in zip(chosen, found):
                member[position] = document
        parts.append(member)
    return [joined(found) for found in zip(*parts)]


def documents(
    view_type: tuple[str, str],
    galaxies: np.ndarray,
    rng: np.random.Generator,
    pool: Executor,
) -> list:
    kind, name = view_type
    if kind == "combination":
        return combined(VIEW_COMBINATIONS[name], galaxies, rng, pool)
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
        embeddings[members] = text_model().encode(
            documents(VIEW_TYPES[view_type], owners[members], rng, pool),
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


def generate_text_views(parts: int) -> None:
    folder = artifact("text_views")
    folder.mkdir(exist_ok=True)
    galaxies = np.setdiff1d(np.arange(len(dataset())), held_out())
    found = weights(galaxies)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(PLOTTERS, mp_context=context) as pool:
        for index in tqdm(range(parts), desc="text views"):
            path = folder / f"part-{index:05d}.parquet"
            if not path.exists():
                hidden = folder / f".{path.name}"
                pq.write_table(part(index, galaxies, found, pool), hidden)
                hidden.rename(path)
