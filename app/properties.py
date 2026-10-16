import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

from . import pql, templates
from .config import LS, PROBED, PROPERTIES, PROVABGS, REDSHIFT, SEED, artifact
from .observations import column_values, measured_redshifts

ALPHAS = np.logspace(-3, 4, 15)
FOLDS = KFold(shuffle=True, random_state=SEED)


def provabgs(galaxies: np.ndarray) -> dict[str, np.ndarray]:
    mass = column_values(f"LOG_MSTAR{PROVABGS}", galaxies)
    with np.errstate(divide="ignore", invalid="ignore"):
        sfr = np.log10(column_values(f"AVG_SFR{PROVABGS}", galaxies))
    return dict(zip(PROBED, (mass, sfr, sfr - mass)))


def colours(galaxies: np.ndarray) -> np.ndarray:
    green, red = (column_values(f"FLUX_{band}{LS}", galaxies) for band in "GR")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((green > 0) & (red > 0), -2.5 * np.log10(green / red), np.nan)


def redshift_widths() -> np.ndarray:
    bins = templates.bin_values(REDSHIFT)
    return np.concatenate(
        [
            templates.half_widths(pql.table_value_probabilities(rows, REDSHIFT), bins)
            for rows in pql.predictions().select([REDSHIFT]).to_batches()
        ]
    )[:, 0]


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
    columns = {
        f"{REDSHIFT}_measured": measured_redshifts(galaxies)[0],
        f"{REDSHIFT}_predicted": np.load(artifact(f"template_{REDSHIFT}"))[:, 0],
        f"{REDSHIFT}_width": redshift_widths(),
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
        name: pa.array(values, mask=~np.isfinite(values))
        for name, values in columns.items()
    }
    table = pa.table({"galaxy": galaxies} | arrays, schema=PROPERTIES)
    pq.write_table(table, artifact("properties"))
