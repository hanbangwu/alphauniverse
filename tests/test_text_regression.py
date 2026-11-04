import importlib
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app import embeddings, text_views
from app.config import (
    ARTIFACTS,
    GEMMA_DIM,
    GRAM_NAMES,
    MODES,
    TEXT_VIEWS,
    TEXT_VIEWS_HELD_OUT,
    VIEW_COMBINATIONS,
    VIEW_TYPES,
    galaxy_count,
)

torch = pytest.importorskip("torch")
text_regression = importlib.import_module("app.text_regression")


def identity_bases(width: int) -> dict[str, torch.Tensor]:
    return {
        "components": torch.eye(width).expand(len(VIEW_TYPES), -1, -1),
        "centre": torch.zeros(len(VIEW_TYPES), width),
        "scale": torch.ones(len(VIEW_TYPES), width),
        "offset": torch.zeros(len(GRAM_NAMES)),
        "projection": torch.eye(len(GRAM_NAMES))[:, :width],
    }


def test_identification_is_one_for_predictions_matching_their_own_views_and_a_half_for_alike_ones() -> (
    None
):
    held = text_regression.HeldOut(
        torch.arange(3),
        torch.eye(len(GRAM_NAMES))[:3],
        torch.arange(3),
        torch.zeros(3, dtype=torch.long),
        torch.eye(3),
    )
    present = torch.ones(1, len(GRAM_NAMES))

    separate = text_regression.identification(
        lambda condition, present, kind: condition, held, identity_bases(3), present
    )
    same = text_regression.identification(
        lambda condition, present, kind: torch.ones_like(condition),
        held,
        identity_bases(3),
        present,
    )

    assert separate[0].item() == 1
    assert same[0].item() == 0.5
    assert torch.isnan(separate[1:]).all()


def test_a_prediction_s_coefficients_are_what_the_network_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_regression, "TEXT_RANK", 3)
    generator = torch.Generator().manual_seed(0)
    kinds = len(VIEW_TYPES)
    components = torch.linalg.qr(torch.randn(kinds, 12, 3, generator=generator)).Q
    bases = identity_bases(3) | {
        "components": components.transpose(1, 2),
        "centre": torch.randn(kinds, 12, generator=generator),
        "scale": torch.rand(kinds, 3, generator=generator) + 0.5,
    }
    rows = torch.randn(4, len(GRAM_NAMES), generator=generator)
    present = torch.ones(4, len(GRAM_NAMES))

    def network(condition, present, kind):
        return condition + kind[:, None]

    found = text_regression.predicted(network, bases, rows, present)

    conditions = text_regression.condition(rows, present, bases)
    for kind in (0, kinds - 1):
        view_types = torch.full((4,), kind)
        torch.testing.assert_close(
            text_regression.coefficients(bases, view_types, found[kind]),
            conditions + kind,
        )


def test_a_masked_block_does_not_reach_the_condition() -> None:
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(4, len(GRAM_NAMES) * 3, generator=generator)
    bases = {
        "offset": torch.randn(rows.shape[1]),
        "projection": torch.randn(rows.shape[1], 5),
    }
    present = torch.ones(4, len(GRAM_NAMES))
    present[:, 2] = 0
    changed = rows.clone()
    changed[:, 6:9] = torch.randn(4, 3, generator=generator)

    torch.testing.assert_close(
        text_regression.condition(rows, present, bases),
        text_regression.condition(changed, present, bases),
    )


def test_a_view_type_too_rare_for_its_own_basis_gets_its_mode_s() -> None:
    counts = torch.full((len(VIEW_TYPES),), 200)
    rare = VIEW_TYPES.index(("observed", "sdss_spectrum"))
    counts[rare] = 3
    view_types = torch.repeat_interleave(torch.arange(len(VIEW_TYPES)), counts)
    embeddings = torch.randn(len(view_types), GEMMA_DIM)

    bases = text_regression.fitted_bases(view_types, embeddings)

    components = bases["components"][rare]
    torch.testing.assert_close(
        components @ components.T,
        torch.eye(text_regression.TEXT_RANK),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        bases["mean"][rare], embeddings[view_types == rare].mean(dim=0)
    )


def test_training_draws_every_view_type_a_galaxy_has_equally_often(
    tree: Path,
) -> None:
    drawn = text_views.weights(np.arange(galaxy_count()))

    found = (
        drawn * text_regression.training_weights(torch.arange(len(VIEW_TYPES))).numpy()
    )

    np.testing.assert_allclose(found[drawn > 0], 1)


def written_views(
    path: Path, galaxies: np.ndarray, rng: np.random.Generator, index: int
) -> None:
    view_types = np.tile(np.arange(len(VIEW_TYPES)), 2)
    vectors = rng.standard_normal((len(view_types), GEMMA_DIM))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pq.write_table(
        pa.table(
            {
                "galaxy": rng.choice(galaxies, len(view_types)),
                "view_type": view_types,
                "round": np.full(len(view_types), index // 2),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    vectors.astype(np.float16).ravel(), GEMMA_DIM
                ),
            },
            schema=TEXT_VIEWS,
        ),
        path,
    )


def test_training_uses_every_part_for_a_fixed_number_of_steps_and_keeps_the_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    rng = np.random.default_rng(0)
    held = np.array([1, 5, 9])
    training = np.setdiff1d(np.arange(12), held)
    joint = rng.standard_normal((12, len(GRAM_NAMES) * 2)).astype(np.float32)
    np.save(tmp_path / "joint_embedding.npy", joint)
    folder = tmp_path / "text_views.parquet"
    folder.mkdir()
    owners, view_types = np.divmod(
        np.arange(len(held) * len(VIEW_TYPES)), len(VIEW_TYPES)
    )
    vectors = rng.standard_normal((len(owners), GEMMA_DIM)).astype(np.float16)
    pq.write_table(
        pa.table(
            {
                "galaxy": held[owners],
                "view_type": view_types,
                "embedding": pa.FixedSizeListArray.from_arrays(
                    vectors.ravel(), GEMMA_DIM
                ),
            },
            schema=TEXT_VIEWS_HELD_OUT,
        ),
        tmp_path / "text_views_held_out.parquet",
    )
    generated = []

    def generate_text_views(parts: int) -> None:
        generated.append(parts)
        for index in range(parts):
            written_views(folder / f"part-{index:05d}.parquet", training, rng, index)

    steps = []
    stepped = text_regression.stepped

    def counted(*args):
        steps.append(text_regression.STEPS)
        return stepped(*args)

    for module in (text_regression, text_views, embeddings):
        monkeypatch.setattr(
            module, "artifact", lambda role: tmp_path / f"{role}.{ARTIFACTS[role]}"
        )
    monkeypatch.setattr(text_views, "training", lambda: training)
    monkeypatch.setattr(text_views, "generate_text_views", generate_text_views)
    monkeypatch.setattr(text_regression, "stepped", counted)
    for name, value in {
        "TEXT_RANK": 3,
        "TEXT_CONDITION": 4,
        "HIDDEN": 8,
        "BATCH": 8,
        "STEPS": 2,
        "TRAINING_STEPS": 6,
        "LOGGED": 1,
        "TEXT_VIEW_PARTS": 3,
        "PARTS_PER_ROUND": 2,
    }.items():
        monkeypatch.setattr(text_regression, name, value)

    text_regression.train_text_regression()

    printed = capsys.readouterr().out
    report = json.loads(printed[printed.find("{\n") :])
    stored = np.load(tmp_path / "text_bases.npz")
    assert generated == [3]
    assert sum(steps) == 6
    assert report["views"] == 3 * 2 * len(VIEW_TYPES)
    assert stored["components"].shape == (len(VIEW_TYPES), 3, GEMMA_DIM)
    assert stored["projection"].shape == (len(GRAM_NAMES) * 2, 4)


def test_no_component_s_scale_falls_below_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_regression, "TEXT_RANK", 4)
    rng = np.random.default_rng(0)
    view_types = torch.arange(len(VIEW_TYPES)).repeat(200)
    found = torch.from_numpy(rng.standard_normal((len(view_types), 6))).float()
    found[:, 3:] *= 1e-6
    found = torch.nn.functional.pad(found, (0, GEMMA_DIM - 6))
    scale = text_regression.fitted_bases(view_types, found)["scale"]
    median = scale.median(dim=1, keepdim=True).values
    assert bool((scale >= text_regression.SCALE_FLOOR * median - 1e-6).all())


def test_a_galaxy_s_text_vectors_do_not_depend_on_its_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "TEXT_RANK": 3,
        "TEXT_CONDITION": 4,
        "HIDDEN": 8,
        "GALAXIES": 2,
    }.items():
        monkeypatch.setattr(text_regression, name, value)
    for module in (text_regression, embeddings):
        monkeypatch.setattr(
            module, "artifact", lambda role: tmp_path / f"{role}.{ARTIFACTS[role]}"
        )
    rng = np.random.default_rng(0)
    kinds = len(VIEW_TYPES)
    width = len(GRAM_NAMES) * 2
    np.savez(
        tmp_path / "text_bases.npz",
        components=rng.standard_normal((kinds, 3, GEMMA_DIM)).astype(np.float32),
        centre=rng.standard_normal((kinds, GEMMA_DIM)).astype(np.float32),
        scale=rng.uniform(0.5, 1.5, (kinds, 3)).astype(np.float32),
        offset=rng.standard_normal(width).astype(np.float32),
        projection=rng.standard_normal((width, 4)).astype(np.float32),
    )
    torch.save(
        text_regression.Regression().state_dict(), tmp_path / "text_regression.pt"
    )
    np.save(
        tmp_path / "joint_embedding.npy",
        rng.standard_normal((5, width)).astype(np.float32),
    )
    queries = rng.standard_normal((4, GEMMA_DIM)).astype(np.float32)
    monkeypatch.setattr(text_regression, "generic_queries", lambda: ["a"] * 4)
    monkeypatch.setattr(text_regression, "embed_queries", lambda _: queries)

    text_regression.generate_text_vectors()

    stored = np.load(tmp_path / "text_vectors.npy")
    network, bases = text_regression.stored_regression()
    joint = embeddings.loaded("joint_embedding", torch.float32)
    alone = text_regression.predicted_vectors(network, bases, joint, torch.tensor([3]))
    assert stored.shape == (5, len(MODES), GEMMA_DIM)
    np.testing.assert_allclose(stored[3], alone[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.load(tmp_path / "text_query_mean.npy"), queries.mean(axis=0)
    )


def test_a_galaxy_s_mode_vectors_sum_to_the_mean_of_its_predictions() -> None:
    predictions = torch.randn(
        len(VIEW_TYPES), 2, 12, generator=torch.Generator().manual_seed(0)
    )

    found = text_regression.mode_vectors(predictions)

    torch.testing.assert_close(found.sum(dim=1), predictions.mean(dim=0))


def test_mode_weights_give_each_mode_its_shapley_value() -> None:
    contributions = torch.randn(
        len(VIEW_TYPES), generator=torch.Generator().manual_seed(0)
    )
    required = [set(VIEW_COMBINATIONS.get(name, (name,))) for _, name in VIEW_TYPES]

    def value(modes: set[str]) -> float:
        kinds = [kind for kind, needed in enumerate(required) if needed <= modes]
        return float(contributions[kinds].sum()) / len(VIEW_TYPES)

    expected = []
    for mode in MODES:
        others = [other for other in MODES if other != mode]
        expected.append(
            sum(
                math.factorial(size)
                * math.factorial(len(MODES) - size - 1)
                / math.factorial(len(MODES))
                * (value({*chosen, mode}) - value(set(chosen)))
                for size in range(len(MODES))
                for chosen in combinations(others, size)
            )
        )

    found = contributions @ text_regression.mode_weights()

    np.testing.assert_allclose(found.numpy(), expected, rtol=1e-5)
