from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from app import observations, observed_grams, pql, templates
from app.config import (
    BANDS,
    CROP_PIXELS,
    GRAM_NAMES,
    IMAGE_MODES,
    MODES,
    N_IMAGE_TOKENS,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    SPECTRUM_MODES,
    TABLE_MODES,
    TABLE_VALUE_SURVEYS,
    TOKEN_GRAMS,
    TOKEN_SURVEYS,
    artifact,
)
from app.observations import MAGNITUDE_ERRORS
from scripts.fixture import covered

MODE = "hsc_image"
GALAXY = 7
WIDTH = len(BANDS["hsc"]) * CROP_PIXELS**2


def random_data(
    rng: np.random.Generator, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.from_numpy(rng.normal(size=(count, WIDTH)))
    information = torch.from_numpy(rng.random((count, WIDTH)))
    return values, information


def test_same_place_scores_are_chi_square_differences_from_the_reference() -> None:
    rng = np.random.default_rng(0)
    queries = random_data(rng, 3)[0].float()
    values, information = random_data(rng, 4)
    reference = queries.mean(dim=0)

    def chi_square(model: torch.Tensor, free: bool) -> torch.Tensor:
        if free:
            amplitude = (information * values * model).sum(-1) / (
                information * model**2
            ).sum(-1)
            model = amplitude[..., None] * model
        return (information * (values - model) ** 2).sum(-1)

    for free in (False, True):
        found = observed_grams.same_place(
            queries, reference, information * values, information, free
        )
        saturation = observed_grams.saturation(
            reference,
            information * values,
            information,
            (information * values**2).sum(-1),
            free,
        )

        null = chi_square(reference.double(), free)
        expected = 0.5 * (null - chi_square(queries.double()[:, None], free))
        np.testing.assert_allclose(found, expected, rtol=1e-9, err_msg=str(free))
        np.testing.assert_allclose(saturation, 0.5 * null, rtol=1e-9, err_msg=str(free))


def slot_scores(
    query: torch.Tensor,
    reference: torch.Tensor,
    values: torch.Tensor,
    information: torch.Tensor,
) -> torch.Tensor:
    query, reference, values, information = (
        observed_grams.slotted(part[None], MODE)[0]
        for part in (query, reference, values, information)
    )
    scores = (
        query @ (information * values).T
        - 0.5 * query**2 @ information.T
        - (
            (information * values * reference).sum(-1)
            - 0.5 * (information * reference**2).sum(-1)
        )
    )
    scores[:, information.sum(-1) == 0] = -torch.inf
    return scores


def test_any_position_takes_each_query_slot_s_best_gallery_slot_with_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observed_grams, "ROWS", 2 * N_IMAGE_TOKENS)
    rng = np.random.default_rng(1)
    queries = random_data(rng, 3)[0].float()
    values, information = random_data(rng, 5)
    information[1].view(-1, CROP_PIXELS, CROP_PIXELS)[:, : CROP_PIXELS // 2] = 0
    reference = queries.mean(dim=0)
    kept = torch.ones(N_IMAGE_TOKENS, dtype=torch.bool)

    found = observed_grams.anywhere(
        *observed_grams.data_features(
            queries, reference, information * values, information, MODE, kept
        ),
        symmetric=False,
    )

    for query in range(len(queries)):
        for gallery in range(len(values)):
            expected = slot_scores(
                queries[query].double(),
                reference.double(),
                values[gallery],
                information[gallery],
            )
            np.testing.assert_allclose(
                found[query, gallery], expected.amax(dim=1).sum(), rtol=1e-4
            )


def test_template_tiles_give_both_directions_of_the_nearest_patch_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observed_grams, "ROWS", 2 * N_IMAGE_TOKENS)
    scaled = torch.from_numpy(np.random.default_rng(2).normal(size=(5, WIDTH))).float()
    kept = torch.ones(N_IMAGE_TOKENS, dtype=torch.bool)

    found = observed_grams.anywhere(
        *observed_grams.template_features(scaled, MODE, kept), symmetric=True
    )

    patches = observed_grams.slotted(scaled.double(), MODE)
    for query in range(len(scaled)):
        for gallery in range(len(scaled)):
            distances = torch.cdist(patches[query], patches[gallery]) ** 2
            np.testing.assert_allclose(
                found[query, gallery],
                (-0.5 * distances).amax(dim=1).sum(),
                rtol=1e-4,
                atol=1e-2,
            )


def image(rng: np.random.Generator, survey: str) -> dict:
    size = CROP_PIXELS + 8
    shape = (len(BANDS[survey]), size, size)
    return {
        "band": [band.lower() for band in BANDS[survey]],
        "flux": rng.normal(size=shape).astype(np.float32),
        "ivar": rng.random(shape).astype(np.float32),
    }


def spectrum(rng: np.random.Generator, wavelength: np.ndarray) -> dict:
    return {
        "lambda": wavelength,
        "flux": rng.normal(1, 0.1, len(wavelength)),
        "ivar": rng.random(len(wavelength)),
        "mask": np.zeros(len(wavelength), dtype=bool),
    }


def dataset(rng: np.random.Generator) -> Dataset:
    galaxies = range(12)
    wavelengths = {
        "desi": 3600 + 0.8 * np.arange(7781),
        "sdss": np.geomspace(3800, 9200, 3853),
    }
    return Dataset.from_dict(
        {
            TOKEN_SURVEYS["ls"]: [image(rng, "ls") for _ in galaxies],
            TOKEN_SURVEYS["hsc"]: [
                image(rng, "hsc") if covered("hsc", galaxy) else None
                for galaxy in galaxies
            ],
            **{
                TOKEN_SURVEYS[survey]: [
                    spectrum(rng, wavelengths[survey])
                    if covered(survey, galaxy)
                    else None
                    for galaxy in galaxies
                ]
                for survey in wavelengths
            },
            **{
                column: [
                    value if covered(REDSHIFT, galaxy) else None for galaxy in galaxies
                ]
                for column, value in zip(REDSHIFT_COLUMNS["desi"], (0.5, 1e-4, False))
            },
            **{column: [None] * len(galaxies) for column in REDSHIFT_COLUMNS["sdss"]},
            **{
                column: rng.lognormal(size=len(galaxies)).tolist()
                for columns in TABLE_VALUE_SURVEYS.values()
                for column in columns
            },
            **{
                column: rng.uniform(0.01, 0.1, len(galaxies)).tolist()
                for column in MAGNITUDE_ERRORS.values()
            },
        }
    )


def test_stored_matrices_score_every_template_against_each_observed_galaxy(
    tree: Path, random_weights: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = dataset(np.random.default_rng(3))
    monkeypatch.setattr(observations, "dataset", lambda: rows)
    monkeypatch.setattr(torch.backends.cudnn.conv, "fp32_precision", "ieee")
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    templates.generate_templates()

    observed_grams.generate_observed_grams()

    references = np.load(artifact("observed_reference"))
    for mode in MODES:
        free = mode in SPECTRUM_MODES
        galaxies = observations.observed(mode)
        queries = torch.from_numpy(np.load(artifact(f"template_{mode}")))
        queries = queries.reshape(len(queries), -1)
        reference = torch.from_numpy(references[f"{mode}_reference"])
        weighted, information, squares = (
            torch.from_numpy(part).double()
            for part in observations.measured(mode, galaxies)
        )
        same = np.load(artifact(f"data_gram_{mode}"))
        np.testing.assert_array_equal(references[f"{mode}_galleries"], galaxies)
        np.testing.assert_allclose(
            same,
            observed_grams.same_place(queries, reference, weighted, information, free),
            rtol=1e-5,
            err_msg=mode,
        )
        np.testing.assert_allclose(
            references[f"{mode}_saturation"],
            observed_grams.saturation(reference, weighted, information, squares, free),
            rtol=1e-9,
            err_msg=mode,
        )
        if free:
            amplitudes = torch.from_numpy(references[f"{mode}_amplitudes"])[:, None]
            weighted, information = weighted * amplitudes, information * amplitudes**2
        depth = information.median(dim=0).values.sqrt()
        np.testing.assert_allclose(
            references[f"{mode}_depth"], depth, rtol=1e-6, err_msg=mode
        )
        scaled = queries * torch.from_numpy(references[f"{mode}_depth"])
        if free:
            scaled /= scaled.norm(dim=1, keepdim=True)
        np.testing.assert_allclose(
            np.load(artifact(f"template_gram_{mode}")),
            -0.5 * torch.cdist(scaled.double(), scaled.double()) ** 2,
            rtol=1e-4,
            atol=1e-3,
            err_msg=mode,
        )
        if len(TOKEN_GRAMS[mode]) == 1:
            continue
        name = f"{mode}_anywhere"
        kept = observed_grams.slotted(depth[None], mode)[0].sum(-1) > 0
        anywhere = np.load(artifact(f"data_gram_{name}"))
        np.testing.assert_allclose(
            anywhere,
            observed_grams.anywhere(
                *observed_grams.data_features(
                    queries, reference, weighted, information, mode, kept
                ),
                symmetric=False,
            ),
            rtol=1e-5,
            err_msg=name,
        )
        if not free:
            assert (anywhere >= same - 1e-3 * (1 + np.abs(same))).all(), name
        np.testing.assert_allclose(
            np.load(artifact(f"template_gram_{name}")),
            observed_grams.anywhere(
                *observed_grams.template_features(scaled, mode, kept),
                symmetric=True,
            ),
            rtol=1e-5,
            atol=1e-5,
            err_msg=name,
        )


def test_template_embeddings_decompose_each_template_gram_with_widths(
    tree: Path, random_weights: None
) -> None:
    rng = np.random.default_rng(3)
    count = pql.predictions().num_rows
    for name in GRAM_NAMES:
        np.save(
            artifact(f"template_gram_{name}"),
            rng.normal(size=(count, count)).astype(np.float32),
        )
    depths = {
        **{mode: rng.random(values) for mode, (_, values, _) in TABLE_MODES.items()},
        **{mode: rng.random(1) for mode in (*IMAGE_MODES, *SPECTRUM_MODES)},
    }
    np.savez(
        artifact("observed_reference"),
        **{f"{mode}_depth": depth for mode, depth in depths.items()},
    )

    observed_grams.generate_template_embeddings()

    expected = templates.widths(pql.row(GALAXY), depths)
    for mode, names in TOKEN_GRAMS.items():
        for name in names:
            gram = np.load(artifact(f"template_gram_{name}")).astype(np.float64)
            found = np.load(artifact(f"template_embedding_{name}"))
            difference = (
                found["queries"] @ found["galleries"].T + found["offsets"] - gram
            )
            np.testing.assert_allclose(
                difference - difference[:, :1], 0, atol=1e-3, err_msg=name
            )
            np.testing.assert_allclose(
                found["widths"][GALAXY], expected[mode][0], rtol=1e-6, err_msg=name
            )
