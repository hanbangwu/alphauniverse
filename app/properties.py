import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

from . import pql, templates
from .config import (
    ANCHOR,
    LS,
    LS_PROPERTIES,
    PROPERTIES,
    PROVABGS,
    PROVABGS_PROPERTIES,
    REDSHIFT,
    SEED,
    SPECTRAL_PROPERTIES,
    SPECTRUM_SURVEYS,
    ZERO_POINTS,
    artifact,
)
from .observations import column_values, measured_redshifts, samples, spectra

ALPHAS = np.logspace(-3, 4, 15)
FOLDS = KFold(shuffle=True, random_state=SEED)
HALPHA = 6564.61
HALPHA_BANDS = ((6490.0, 6530.0), (6600.0, 6640.0), (HALPHA - 10, HALPHA + 10))
BREAK_BANDS = ((3850.0, 3950.0), (4000.0, 4100.0))
MINIMUM_PIXELS = 5
MINIMUM_SIGNAL = 3
BATCH = 256


def provabgs(galaxies: np.ndarray) -> dict[str, np.ndarray]:
    mass = column_values(f"LOG_MSTAR{PROVABGS}", galaxies)
    with np.errstate(divide="ignore", invalid="ignore"):
        sfr = np.log10(column_values(f"AVG_SFR{PROVABGS}", galaxies))
    return dict(zip(PROVABGS_PROPERTIES, (mass, sfr, sfr - mass)))


def magnitudes(galaxies: np.ndarray, band: str) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = column_values(f"FLUX_{band}{LS}", galaxies)
        return ZERO_POINTS[ANCHOR] - 2.5 * np.log10(flux)


def colours(galaxies: np.ndarray) -> np.ndarray:
    return magnitudes(galaxies, "G") - magnitudes(galaxies, "R")


def ls_properties(galaxies: np.ndarray) -> dict[str, np.ndarray]:
    radius, first, second = (
        column_values(f"{name}{LS}", galaxies)
        for name in ("SHAPE_R", "SHAPE_E1", "SHAPE_E2")
    )
    ellipticity = np.hypot(first, second)
    axis_ratio = np.where(
        ellipticity > 0, (1 - ellipticity) / (1 + ellipticity), np.nan
    )
    return dict(zip(LS_PROPERTIES, (radius, magnitudes(galaxies, "R"), axis_ratio)))


def band(rest: np.ndarray, low: float, high: float) -> np.ndarray | None:
    inside = (rest >= low) & (rest <= high)
    if inside.sum() < MINIMUM_PIXELS or rest[0] > low or rest[-1] < high:
        return None
    return inside


def halpha_ew(rest: np.ndarray, flux: np.ndarray) -> float:
    bands = [band(rest, *edges) for edges in HALPHA_BANDS]
    if any(inside is None for inside in bands):
        return np.nan
    blue, red, window = bands
    continuum = (np.median(flux[blue]) + np.median(flux[red])) / 2
    if continuum <= 0:
        return np.nan
    return np.trapezoid(flux[window] / continuum - 1, rest[window])


def dn4000(rest: np.ndarray, flux: np.ndarray, ivar: np.ndarray) -> float:
    bands = [band(rest, *edges) for edges in BREAK_BANDS]
    if any(inside is None for inside in bands):
        return np.nan
    blue, red = (np.mean(flux[inside] * rest[inside] ** 2) for inside in bands)
    error = np.sqrt(np.sum(rest[bands[0]] ** 4 / ivar[bands[0]])) / bands[0].sum()
    return red / blue if blue > MINIMUM_SIGNAL * error else np.nan


def spectral_properties(
    galaxies: np.ndarray, redshifts: np.ndarray, surveys: list[str | None]
) -> dict[str, np.ndarray]:
    found = np.full((len(SPECTRAL_PROPERTIES), len(galaxies)), np.nan)
    for survey in SPECTRUM_SURVEYS:
        chosen = np.flatnonzero(np.array(surveys) == survey)
        for start in range(0, len(chosen), BATCH):
            part = chosen[start : start + BATCH]
            cells = spectra(survey, galaxies[part])
            for row, position in enumerate(part):
                if not cells[row].is_valid:
                    continue
                wavelength, flux, ivar = samples(cells, row)
                rest = wavelength / (1 + redshifts[position])
                found[:, position] = halpha_ew(rest, flux), dn4000(rest, flux, ivar)
    return dict(zip(SPECTRAL_PROPERTIES, found))


def spread_quantiles(
    probabilities: np.ndarray, values: np.ndarray, share: float
) -> np.ndarray:
    weights = np.where(np.isnan(values), 0, probabilities)
    cumulative = weights.cumsum(axis=1)
    target = share * cumulative[:, -1]
    index = (cumulative < target[:, None]).sum(axis=1)
    rows = np.arange(len(weights))
    fraction = 1 - (cumulative[rows, index] - target) / weights[rows, index]
    lower = values[np.maximum(index - 1, 0)]
    return lower + fraction * (values[index] - lower)


def predicted_redshifts() -> tuple[np.ndarray, np.ndarray]:
    values = templates.bin_values(REDSHIFT)[0]
    batches = [
        pql.table_value_probabilities(rows, REDSHIFT)[:, 0]
        for rows in pql.predictions().select([REDSHIFT]).to_batches()
    ]
    lower, median, upper = (
        np.concatenate([spread_quantiles(found, values, share) for found in batches])
        for share in (templates.CENTRAL[0], 0.5, templates.CENTRAL[1])
    )
    return median, (upper - lower) / 2


def probed(features: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    known = np.isfinite(values)
    labelled, targets = features[known], values[known]
    model = RidgeCV(alphas=ALPHAS)
    held_out = cross_val_predict(model, labelled, targets, cv=FOLDS)
    predicted = model.fit(labelled, targets).predict(features)
    predicted[known] = held_out
    return predicted, root_mean_squared_error(targets, held_out)


def generate_properties() -> None:
    features = np.load(artifact("joint_embedding"))
    galaxies = np.arange(len(features))
    redshifts, surveys = measured_redshifts(galaxies)
    predicted, width = predicted_redshifts()
    columns = {
        f"{REDSHIFT}_measured": redshifts,
        f"{REDSHIFT}_predicted": predicted,
        f"{REDSHIFT}_width": width,
        "g_r": colours(galaxies),
    }
    measured = (
        provabgs(galaxies)
        | ls_properties(galaxies)
        | spectral_properties(galaxies, redshifts, surveys)
    )
    for label, values in measured.items():
        predicted, width = probed(features, values)
        columns |= {
            f"{label}_measured": values,
            f"{label}_predicted": predicted,
            f"{label}_width": np.full(len(galaxies), width),
        }
    arrays = {
        name: pa.array(values, mask=~np.isfinite(values))
        for name, values in columns.items()
    }
    table = pa.table({"galaxy": galaxies} | arrays, schema=PROPERTIES)
    pq.write_table(table, artifact("properties"))
