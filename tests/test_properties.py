from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from app import observations, properties, templates
from app.config import (
    LS,
    PROBED,
    PROPERTIES,
    PROVABGS,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    REDSHIFT_VOCABULARY,
    artifact,
)

LABELLED = 30


def test_a_labelled_galaxy_s_prediction_does_not_depend_on_its_own_label() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 6))
    values = features @ rng.normal(size=6) + rng.normal(scale=0.1, size=40)
    values[LABELLED:] = np.nan
    changed = values.copy()
    changed[3] += 10

    predicted, width = properties.probed(features, values)
    moved, _ = properties.probed(features, changed)

    assert moved[3] == predicted[3]
    residuals = predicted[:LABELLED] - values[:LABELLED]
    assert width == pytest.approx(np.sqrt(np.mean(residuals**2)))


def test_a_redshift_bin_s_probability_spreads_from_the_bin_below_to_its_value() -> None:
    values = np.array([0.0, 0.1, 0.2, 0.3, np.nan], np.float32)
    probabilities = np.array(
        [[0, 0, 1, 0, 0], [0, 0, 0.5, 0.5, 0], [0, 0, 1, 0, 5]], np.float32
    )

    lower, upper = (
        properties.spread_quantiles(probabilities, values, share)
        for share in templates.CENTRAL
    )

    np.testing.assert_allclose(lower, [0.116, 0.132, 0.116], rtol=1e-6)
    np.testing.assert_allclose(upper, [0.184, 0.268, 0.184], rtol=1e-6)


def test_properties_hold_each_measured_value_and_a_prediction_for_every_galaxy(
    tree: Path, galaxies: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(1)
    labelled = np.arange(galaxies) % 4 != 0
    redshift = np.where(np.arange(galaxies) % 3 == 0, 0.5, np.nan)
    green, red = rng.lognormal(size=galaxies), rng.lognormal(size=galaxies)
    red[2] = -1
    value, error, warning = REDSHIFT_COLUMNS["desi"]
    rows = Dataset.from_dict(
        {
            value: [None if np.isnan(z) else z for z in redshift],
            error: [1e-4] * galaxies,
            warning: [False] * galaxies,
            **{column: [None] * galaxies for column in REDSHIFT_COLUMNS["sdss"]},
            f"FLUX_G{LS}": green.tolist(),
            f"FLUX_R{LS}": red.tolist(),
            f"LOG_MSTAR{PROVABGS}": [
                float(mass) if kept else None
                for mass, kept in zip(rng.normal(10, 1, galaxies), labelled)
            ],
            f"AVG_SFR{PROVABGS}": [
                float(sfr) if kept else None
                for sfr, kept in zip(rng.lognormal(size=galaxies), labelled)
            ],
        }
    )
    monkeypatch.setattr(observations, "dataset", lambda: rows)
    bins = {REDSHIFT: np.linspace(0.01, 6, REDSHIFT_VOCABULARY, dtype=np.float32)[None]}
    monkeypatch.setattr(templates, "bin_values", bins.get)
    np.save(
        artifact("joint_embedding"), rng.normal(size=(galaxies, 4)).astype(np.float32)
    )
    template = rng.normal(size=(galaxies, 1)).astype(np.float32)
    np.save(artifact(f"template_{REDSHIFT}"), template)

    properties.generate_properties()

    table = pq.read_table(artifact("properties"))
    assert table.schema == PROPERTIES
    found = {name: table.column(name).to_numpy() for name in PROPERTIES.names}
    np.testing.assert_array_equal(
        found[f"{REDSHIFT}_measured"], redshift.astype(np.float32)
    )
    np.testing.assert_array_equal(np.isfinite(found["mass_measured"]), labelled)
    np.testing.assert_array_equal(found[f"{REDSHIFT}_predicted"], template[:, 0])
    for label in PROBED:
        assert np.isfinite(found[f"{label}_predicted"]).all(), label
    expected = -2.5 * np.log10(green / np.abs(red))
    expected[2] = np.nan
    np.testing.assert_allclose(found["g_r"], expected.astype(np.float32), rtol=1e-6)
