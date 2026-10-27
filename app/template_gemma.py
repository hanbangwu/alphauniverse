from io import BytesIO

import numpy as np
import PIL.Image
from numpy.lib.format import open_memmap
from tqdm import tqdm

from .config import (
    GEMMA_DIM,
    IMAGE_MODES,
    MODE_SURVEYS,
    MODES,
    PLOT_INCHES,
    REDSHIFT,
    SPECTRUM_GRID,
    SPECTRUM_MODES,
    TABLE_VALUE_SURVEYS,
    ImageSurvey,
    SpectrumSurvey,
    artifact,
)
from .dataset import predicted_image, smoothed
from .search import templates
from .text_search import text_model

BATCH = 1024
SPECTRUM_RANGES: dict[SpectrumSurvey, tuple[float, float]] = {
    "desi": (3600.0, 9824.0),
    "sdss": (3795.8, 9221.5),
}
FLUX_PERCENTILES = (0.5, 99.5)
FLUX_PADDING = 0.1


def image_document(flux: np.ndarray, survey: ImageSurvey) -> PIL.Image.Image:
    return PIL.Image.open(BytesIO(predicted_image(flux, survey)))


def spectrum_document(
    wavelength: np.ndarray, flux: np.ndarray, survey: SpectrumSurvey
) -> PIL.Image.Image:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    low, high = SPECTRUM_RANGES[survey]
    shown = (wavelength >= low) & (wavelength <= high)
    flux = smoothed(flux)[shown]
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
    return PIL.Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def value_names(mode: str) -> tuple[str, ...]:
    if mode == REDSHIFT:
        return (REDSHIFT,)
    return tuple(
        column.partition("-")[0] for column in TABLE_VALUE_SURVEYS[MODE_SURVEYS[mode]]
    )


def table_document(mode: str, values: np.ndarray) -> str:
    return "\n".join(
        f"{name}: {value:.4g}"
        for name, value in zip(value_names(mode), values, strict=True)
    )


def documents(mode: str, rows: np.ndarray) -> list[PIL.Image.Image] | list[str]:
    if mode in IMAGE_MODES:
        return [image_document(flux, IMAGE_MODES[mode]) for flux in rows]
    if mode in SPECTRUM_MODES:
        return [
            spectrum_document(SPECTRUM_GRID, flux, SPECTRUM_MODES[mode])
            for flux in rows
        ]
    return [table_document(mode, values) for values in rows]


def generate_template_gemma() -> None:
    for mode in MODES:
        found = templates(mode)
        stored = open_memmap(
            artifact(f"template_gemma_{mode}"),
            mode="w+",
            dtype=np.float32,
            shape=(len(found), GEMMA_DIM),
        )
        for start in tqdm(range(0, len(found), BATCH), desc=mode):
            rows = slice(start, start + BATCH)
            stored[rows] = text_model().encode(
                documents(mode, found[rows]), prompt_name="Document"
            )
        stored.flush()
