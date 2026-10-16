import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

from . import pql, templates
from .config import (
    LS,
    PROPERTIES,
    PROVABGS,
    REDSHIFT,
    SEED,
    artifact,
    galaxy_count,
)
from .observations import column_values, measured_redshifts

ALPHAS = np.logspace(-3, 4, 15)
FOLDS = KFold(shuffle=True, random_state=SEED)


def provabgs(galaxies: np.ndarray) -> dict[str, np.ndarray]:
    mass = column_values(f"LOG_MSTAR{PROVABGS}", galaxies)
    with np.errstate(divide="ignore", invalid="ignore"):
        sfr = np.log10(column_values(f"AVG_SFR{PROVABGS}", galaxies))
    return {"mass": mass, "sfr": sfr, "ssfr": sfr - mass}


def colours(galaxies: np.ndarray) -> np.ndarray:
    green, red = (column_values(f"FLUX_{band}{LS}", galaxies) for band in "GR")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((green > 0) & (red > 0), -2.5 * np.log10(green / red), np.nan)


def predicted_redshifts() -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.concatenate(
        [
            pql.table_value_probabilities(rows, REDSHIFT)
            for rows in pql.predictions().select([REDSHIFT]).to_batches()
        ]
    )
    bins = templates.bin_values(REDSHIFT)
    median, lower, upper = (
        templates.quantiles(probabilities, bins, share)[:, 0]
        for share in (0.5, *templates.CENTRAL)
    )
    return median, (upper - lower) / 2


def probed(features: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    known = np.isfinite(values)
    model = RidgeCV(alphas=ALPHAS)
    held_out = cross_val_predict(model, features[known], values[known], cv=FOLDS)
    predicted = model.fit(features[known], values[known]).predict(features)
    predicted[known] = held_out
    return predicted, float(np.sqrt(np.mean((held_out - values[known]) ** 2)))


def generate_properties() -> None:
    galaxies = np.arange(galaxy_count())
    features = np.load(artifact("joint_embedding"))
    predicted, width = predicted_redshifts()
    columns = {
        f"{REDSHIFT}_measured": measured_redshifts(galaxies)[0],
        f"{REDSHIFT}_predicted": predicted,
        f"{REDSHIFT}_width": width,
        "g_r": colours(galaxies),
    }
    for label, values in provabgs(galaxies).items():
        predicted, width = probed(features, values)
        columns |= {
            f"{label}_measured": values,
            f"{label}_predicted": predicted,
            f"{label}_width": np.full(len(galaxies), width),
        }
    arrays = {
        name: pa.array(values.astype(np.float32), mask=~np.isfinite(values))
        for name, values in columns.items()
    }
    table = pa.table({"galaxy": galaxies.astype(np.int32)} | arrays)
    pq.write_table(
        table.select(PROPERTIES.names).cast(PROPERTIES), artifact("properties")
    )
