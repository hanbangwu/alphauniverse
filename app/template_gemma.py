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
FLUX_PERCENTILES = (0.5, 99.9)
FLUX_PADDING = 0.1


def spectrum_document(flux: np.ndarray, survey: SpectrumSurvey) -> PIL.Image.Image:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    low, high = SPECTRUM_RANGES[survey]
    shown = (SPECTRUM_GRID >= low) & (SPECTRUM_GRID <= high)
    flux = smoothed(flux)[shown]
    bottom, top = np.percentile(flux, FLUX_PERCENTILES)
    padding = FLUX_PADDING * (top - bottom)
    figure = Figure(figsize=PLOT_INCHES, layout="tight")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(SPECTRUM_GRID[shown], flux, color="black", linewidth=1)
    axes.set(
        xlim=(low, high),
        ylim=(bottom - padding, top + padding),
        yticks=[],
        xlabel="Observed wavelength (Å)",
        ylabel=r"Flux density $f_\lambda$ (arbitrary units)",
    )
    canvas.draw()
    return PIL.Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")


def documents(mode: str, rows: np.ndarray) -> list[PIL.Image.Image] | list[str]:
    if mode in IMAGE_MODES:
        return [
            PIL.Image.open(BytesIO(predicted_image(flux, IMAGE_MODES[mode])))
            for flux in rows
        ]
    if mode in SPECTRUM_MODES:
        return [spectrum_document(flux, SPECTRUM_MODES[mode]) for flux in rows]
    columns = (
        (REDSHIFT,) if mode == REDSHIFT else TABLE_VALUE_SURVEYS[MODE_SURVEYS[mode]]
    )
    return [
        "\n".join(
            f"{column.partition('-')[0]}: {value:.4g}"
            for column, value in zip(columns, values, strict=True)
        )
        for values in rows
    ]


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
