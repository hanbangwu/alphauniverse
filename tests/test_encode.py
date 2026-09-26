import importlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from app import search as search_module
from app.config import (
    ANCHOR,
    CROP_PIXELS,
    DIM,
    FLAG_SURVEYS,
    N_PATCHES,
    N_SPANS,
    SPECTRUM_SURVEYS,
    STORES,
    TOKEN_SURVEYS,
    artifact,
    device,
    store_schema,
)
from app.search import blocks, source
from scripts.fixture import TOKENS

torch = pytest.importorskip("torch")
encode_module = importlib.import_module("app.encode")

CONFIGS = Path(__file__).parent / "aion"
CACHES = (
    search_module.source,
    search_module.index,
    search_module.with_spectrum,
    search_module.starts,
)


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
    scalars = encode_module.LS_SCALARS + encode_module.HSC_SCALARS
    return {
        TOKEN_SURVEYS[ANCHOR]: image(rng, ["des-g", "des-r", "des-i", "des-z"]),
        TOKEN_SURVEYS["hsc"]: (
            image(rng, ["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"]) if hsc else None
        ),
        TOKEN_SURVEYS["desi"]: spectrum(rng, 7781, 0) if desi else None,
        TOKEN_SURVEYS["sdss"]: spectrum(rng, 3800, 200) if sdss else None,
        **{column: float(rng.random()) + 0.5 for _, column in scalars},
        FLAG_SURVEYS["gz10"]: int(rng.integers(10)),
        FLAG_SURVEYS["provabgs"]: None,
    }


@pytest.fixture(scope="module", autouse=True)
def random_weights() -> Iterator[None]:
    def load(codec_class: type, modality: type) -> object:
        torch.manual_seed(0)
        path = CONFIGS / "codecs" / modality.name / "config.json"
        return codec_class(**json.loads(path.read_text())).eval()

    torch.manual_seed(0)
    config = json.loads((CONFIGS / "config.json").read_text())
    network = encode_module.AION(config | {"encoder_depth": 1, "decoder_depth": 1})
    network = network.to(device()).eval()
    network.requires_grad_(False)
    encode_module.codec.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            encode_module.CodecManager, "_load_codec_from_hf", staticmethod(load)
        )
        patch.setattr(encode_module, "model", lambda: network)
        yield
    encode_module.codec.cache_clear()


def test_each_survey_tokenizes_to_the_fixture_layout() -> None:
    groups = encode_module.tokenize(galaxy(0, hsc=True, desi=True, sdss=True))

    assert {
        survey: sum(slot.shape[1] for slot in group.values())
        for survey, group in groups.items()
    } == TOKENS


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="trailing lambda = -1 padding zeroes the SDSS codec input (#50)",
)
def test_padded_sdss_spectra_keep_their_flux() -> None:
    first, second = (
        encode_module.spectrum(
            encode_module.SDSSSpectrum,
            spectrum(np.random.default_rng(seed), 3800, 200),
        )
        for seed in (1, 2)
    )

    assert not torch.equal(first, second)


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
        first = modality_mask[positions[survey][:N_PATCHES]]
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
    monkeypatch.setenv("ALPHAUNIVERSE_CACHE", str(tmp_path))
    monkeypatch.setattr(encode_module, "dataset", lambda *_: rows)
    for cache in CACHES:
        cache.cache_clear()

    try:
        encode_module.generate_embeddings()
        for cache in CACHES:
            cache.cache_clear()

        for role in STORES:
            assert pq.read_schema(artifact(role)).equals(store_schema(role))
        table = source("encoded").to_table(columns=[ANCHOR, *SPECTRUM_SURVEYS])
        assert blocks(table).shape == (
            len(rows) * N_PATCHES + spectra * N_SPANS,
            DIM,
        )
    finally:
        for cache in CACHES:
            cache.cache_clear()
