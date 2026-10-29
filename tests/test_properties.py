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
    SPECTRUM_SURVEYS,
    artifact,
)

LABELLED = 30
CONTINUUM = 2.0
HALPHA_EW = 30.0
LINE_REDSHIFT = 0.5


def line_spectrum(wavelength: np.ndarray, redshift: float) -> np.ndarray:
    width = 2 * (1 + redshift)
    area = HALPHA_EW * (1 + redshift) * CONTINUUM
    offset = (wavelength - properties.HALPHA * (1 + redshift)) / width
    return CONTINUUM + area / (width * np.sqrt(2 * np.pi)) * np.exp(-(offset**2) / 2)


def flat_break(rest: np.ndarray) -> float:
    blue, red = (
        rest[(rest >= low) & (rest <= high)] for low, high in properties.BREAK_BANDS
    )
    return float(np.mean(red**2) / np.mean(blue**2))


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
    np.testing.assert_allclose(
        properties.spread_quantiles(probabilities, values, 0.5),
        [0.15, 0.2, 0.15],
        rtol=1e-6,
    )


def test_properties_hold_each_measured_value_and_a_prediction_for_every_galaxy(
    tree: Path, galaxies: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(1)
    labelled = np.arange(galaxies) % 4 != 0
    redshift = np.where(np.arange(galaxies) % 2 == 0, LINE_REDSHIFT, np.nan)
    green, red = rng.lognormal(size=galaxies), rng.lognormal(size=galaxies)
    red[2] = -1
    radius = rng.lognormal(size=galaxies)
    first, second = rng.normal(scale=0.2, size=(2, galaxies))
    first[4] = second[4] = 0
    wavelength = np.arange(3500.0, 10500.0)
    spectrum = {
        "lambda": wavelength.tolist(),
        "flux": line_spectrum(wavelength, LINE_REDSHIFT).tolist(),
        "ivar": np.ones_like(wavelength).tolist(),
        "mask": np.zeros(len(wavelength), dtype=bool).tolist(),
    }
    value, error, warning = REDSHIFT_COLUMNS["desi"]
    rows = Dataset.from_dict(
        {
            value: [None if np.isnan(z) else z for z in redshift],
            error: [1e-4] * galaxies,
            warning: [False] * galaxies,
            **{column: [None] * galaxies for column in REDSHIFT_COLUMNS["sdss"]},
            f"FLUX_G{LS}": green.tolist(),
            f"FLUX_R{LS}": red.tolist(),
            f"SHAPE_R{LS}": radius.tolist(),
            f"SHAPE_E1{LS}": first.tolist(),
            f"SHAPE_E2{LS}": second.tolist(),
            SPECTRUM_SURVEYS["desi"]: [
                None if np.isnan(z) else spectrum for z in redshift
            ],
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

    properties.generate_properties()

    table = pq.read_table(artifact("properties"))
    assert table.schema == PROPERTIES
    found = {name: table.column(name).to_numpy() for name in PROPERTIES.names}
    np.testing.assert_array_equal(
        found[f"{REDSHIFT}_measured"], redshift.astype(np.float32)
    )
    np.testing.assert_array_equal(np.isfinite(found["mass_measured"]), labelled)
    predicted = found[f"{REDSHIFT}_predicted"]
    assert (
        (predicted >= bins[REDSHIFT][0, 0]) & (predicted <= bins[REDSHIFT][0, -1])
    ).all()
    for label in PROBED:
        assert np.isfinite(found[f"{label}_predicted"]).all(), label
    expected = -2.5 * np.log10(green / np.abs(red))
    expected[2] = np.nan
    np.testing.assert_allclose(found["g_r"], expected.astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(
        found["radius_measured"], radius.astype(np.float32), rtol=1e-6
    )
    magnitude = 22.5 - 2.5 * np.log10(np.abs(red))
    magnitude[2] = np.nan
    np.testing.assert_allclose(
        found["r_magnitude_measured"], magnitude.astype(np.float32), rtol=1e-6
    )
    ellipticity = np.hypot(first, second)
    axis_ratio = (1 - ellipticity) / (1 + ellipticity)
    axis_ratio[4] = np.nan
    np.testing.assert_allclose(
        found["axis_ratio_measured"], axis_ratio.astype(np.float32), rtol=1e-6
    )
    observed = np.isfinite(redshift)
    np.testing.assert_allclose(
        found["halpha_ew_measured"][observed], HALPHA_EW, rtol=1e-3
    )
    np.testing.assert_allclose(
        found["dn4000_measured"][observed],
        flat_break(wavelength / (1 + LINE_REDSHIFT)),
        rtol=1e-6,
    )
    assert np.isnan(found["dn4000_measured"][~observed]).all()
