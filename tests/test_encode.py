import importlib
from pathlib import Path

import numpy as np
import pytest

from app.config import (
    ANCHOR,
    CROP_PIXELS,
    DIM,
    FLAG_SURVEYS,
    N_IMAGE_TOKENS,
    N_SPECTRUM_TOKENS,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    SPECTRUM_SURVEYS,
    STORES,
    TOKEN_SURVEYS,
    build_dir,
    store_schema,
)
from app.search import N_HSC_TABLE_VALUES, N_LS_TABLE_VALUES, blocks, source
from scripts.fixture import TOKENS, forget

torch = pytest.importorskip("torch")
encode_module = importlib.import_module("app.encode")

pytestmark = pytest.mark.usefixtures("random_weights")


def image(rng: np.random.Generator, bands: list[str]) -> dict[str, list]:
    size = CROP_PIXELS + 8
    return {
        "band": bands,
        "flux": [rng.random((size, size), dtype=np.float32) for _ in bands],
    }


def spectrum(rng: np.random.Generator, samples: int, padding: int) -> dict:
    wavelength = np.concatenate(
        [np.linspace(3800, 9200, samples), np.full(padding, -1.0)]
    )
    return {
        "lambda": wavelength,
        "flux": rng.random(samples + padding),
        "ivar": np.ones(samples + padding),
        "mask": np.zeros(samples + padding, dtype=bool),
    }


def galaxy(seed: int, *, hsc: bool, desi: bool, sdss: bool) -> dict:
    rng = np.random.default_rng(seed)
    table_values = encode_module.LS_TABLE_VALUES + encode_module.HSC_TABLE_VALUES
    present = {"desi": desi, "sdss": sdss}
    redshifts = {
        column: value if present[survey] else None
        for (survey, (redshift, error, warning)), measured in zip(
            REDSHIFT_COLUMNS.items(), (0.5, 0.3), strict=True
        )
        for column, value in ((redshift, measured), (error, 1e-4), (warning, False))
    }
    return {
        **redshifts,
        TOKEN_SURVEYS[ANCHOR]: image(rng, ["des-g", "des-r", "des-i", "des-z"]),
        TOKEN_SURVEYS["hsc"]: (
            image(rng, ["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"]) if hsc else None
        ),
        TOKEN_SURVEYS["desi"]: spectrum(rng, 7781, 0) if desi else None,
        TOKEN_SURVEYS["sdss"]: spectrum(rng, 3800, 200) if sdss else None,
        **{column: float(rng.random()) + 0.5 for _, column in table_values},
        FLAG_SURVEYS["gz10"]: int(rng.integers(10)),
        FLAG_SURVEYS["provabgs"]: None,
    }


def test_each_survey_tokenizes_to_the_fixture_layout() -> None:
    groups = encode_module.tokenize(galaxy(0, hsc=True, desi=True, sdss=True))

    assert {
        survey: sum(slot.shape[1] for slot in group.values())
        for survey, group in groups.items()
    } == TOKENS


def test_spectrum_padding_does_not_change_its_tokens() -> None:
    padded = spectrum(np.random.default_rng(1), 3800, 200)
    unpadded = {field: values[:3800] for field, values in padded.items()}

    with torch.inference_mode():
        first, second = (
            encode_module.spectrum(encode_module.SDSSSpectrum, row)
            for row in (padded, unpadded)
        )

    assert torch.equal(first, second)


def test_each_survey_lands_in_its_own_cell_images_first() -> None:
    groups = encode_module.tokenize(galaxy(0, hsc=True, desi=True, sdss=True))
    _, _, modality_mask = encode_module.encode(groups)
    positions = encode_module.by_survey(
        np.arange(len(modality_mask)), groups, modality_mask
    )
    identifiers = {
        key: info["id"] for key, info in encode_module.model().modality_info.items()
    }

    for survey, group in groups.items():
        owned = {identifiers[key] for key in group}
        assert set(modality_mask[positions[survey]]) == owned
    for survey, image_key in (
        (ANCHOR, encode_module.LegacySurveyImage.token_key),
        ("hsc", encode_module.HSCImage.token_key),
    ):
        first = modality_mask[positions[survey][:N_IMAGE_TOKENS]]
        assert np.all(first == identifiers[image_key])
    assert sorted(np.concatenate(list(positions.values()))) == list(
        range(len(modality_mask))
    )


def test_generated_stores_have_their_schemas_and_the_index_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        galaxy(0, hsc=True, desi=True, sdss=False),
        galaxy(1, hsc=False, desi=False, sdss=False),
        galaxy(2, hsc=False, desi=False, sdss=True),
    ]
    spectra = sum(
        any(row[TOKEN_SURVEYS[survey]] is not None for survey in SPECTRUM_SURVEYS)
        for row in rows
    )
    hsc = sum(row[TOKEN_SURVEYS["hsc"]] is not None for row in rows)
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    build_dir().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(encode_module, "dataset", lambda *_: rows)
    forget()

    try:
        encode_module.generate_embeddings()

        for role in STORES:
            assert source(role).schema.equals(store_schema(role))
        table = source("encoded").to_table(
            columns=[ANCHOR, "hsc", *SPECTRUM_SURVEYS, REDSHIFT]
        )
        assert blocks(table).shape == (
            len(rows) * (N_IMAGE_TOKENS + N_LS_TABLE_VALUES)
            + spectra * (N_SPECTRUM_TOKENS + 1)
            + hsc * N_HSC_TABLE_VALUES,
            DIM,
        )
    finally:
        forget()
