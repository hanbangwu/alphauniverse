import multiprocessing
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from functools import cache, partial
from io import BytesIO
from pathlib import Path

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
    PARTS_PER_ROUND,
    PLOT_INCHES,
    PROTOTYPE_HELD_OUT,
    REDSHIFT,
    SEED,
    SPECTRUM_GRID,
    SPECTRUM_MODES,
    SPECTRUM_SURVEYS,
    TABLE_MODES,
    TABLE_VALUE_SURVEYS,
    TEXT_VIEWS,
    TEXT_VIEWS_HELD_OUT,
    VIEW_COMBINATIONS,
    VIEW_KINDS,
    VIEW_TYPES,
    VIEWS_PER_PART,
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
PLOTTERS = 7
HELD_OUT_GALAXIES = 500


def held_out() -> np.ndarray:
    galaxies = np.arange(len(dataset()))
    split = alignment.held_out(galaxies).cpu().numpy()
    return galaxies[split | np.isin(galaxies, PROTOTYPE_HELD_OUT)]


def training() -> np.ndarray:
    return np.setdiff1d(np.arange(len(dataset())), held_out())


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


def embedded(
    owners: np.ndarray,
    view_types: np.ndarray,
    rng: np.random.Generator,
    pool: Executor,
) -> pa.FixedSizeListArray:
    embeddings = np.empty((len(owners), GEMMA_DIM), np.float16)
    for view_type in np.unique(view_types):
        members = np.flatnonzero(view_types == view_type)
        embeddings[members] = text_model().encode(
            documents(VIEW_TYPES[view_type], owners[members], rng, pool),
            prompt_name="Document",
        )
    return pa.FixedSizeListArray.from_arrays(embeddings.ravel(), GEMMA_DIM)


def part(
    index: int, galaxies: np.ndarray, found: np.ndarray, pool: Executor
) -> pa.Table:
    rng = np.random.default_rng([SEED, index])
    chosen = rng.choice(
        found.size, size=VIEWS_PER_PART, p=(found / found.sum()).ravel()
    )
    owners, view_types = np.divmod(chosen, len(VIEW_TYPES))
    owners = galaxies[owners]
    return pa.table(
        {
            "galaxy": owners,
            "view_type": view_types,
            "round": np.full(VIEWS_PER_PART, index // PARTS_PER_ROUND),
            "embedding": embedded(owners, view_types, rng, pool),
        },
        schema=TEXT_VIEWS,
    )


def held_out_views(pool: Executor) -> pa.Table:
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(1)[0])
    held = held_out()
    prototype = np.intersect1d(held, PROTOTYPE_HELD_OUT)
    others = np.setdiff1d(held, prototype)
    chosen = rng.choice(others, HELD_OUT_GALAXIES - len(prototype), replace=False)
    galaxies = np.union1d(prototype, chosen)
    owners, view_types = np.nonzero(weights(galaxies))
    owners = galaxies[owners]
    return pa.table(
        {
            "galaxy": owners,
            "view_type": view_types,
            "embedding": embedded(owners, view_types, rng, pool),
        },
        schema=TEXT_VIEWS_HELD_OUT,
    )


def part_path(index: int) -> Path:
    return artifact("text_views") / f"part-{index:05d}.parquet"


def written(path: Path, table: Callable[[], pa.Table]) -> None:
    if not path.exists():
        hidden = path.with_name(f".{path.name}")
        pq.write_table(table(), hidden)
        hidden.rename(path)


def generate_text_views(parts: int) -> None:
    artifact("text_views").mkdir(exist_ok=True)
    galaxies = training()
    found = weights(galaxies)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(PLOTTERS, mp_context=context) as pool:
        written(artifact("text_views_held_out"), partial(held_out_views, pool))
        for index in tqdm(range(parts), desc="text views"):
            written(part_path(index), partial(part, index, galaxies, found, pool))
