import importlib
import json
import math
from copy import deepcopy
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
    TEXT_VIEWS,
    TEXT_VIEWS_HELD_OUT,
    VIEW_TYPES,
)

torch = pytest.importorskip("torch")
text_flow = importlib.import_module("app.text_flow")


def test_a_point_s_cosine_from_its_coefficients_is_that_of_the_document_it_stands_for() -> (
    None
):
    generator = torch.Generator().manual_seed(0)
    components = torch.linalg.qr(torch.randn(12, 4, generator=generator)).Q.T
    centre = torch.randn(12, generator=generator) / 4
    points = torch.randn(3, 5, 4, generator=generator)
    directions = torch.nn.functional.normalize(
        torch.randn(7, 12, generator=generator), dim=-1
    )

    found = text_flow.cosines(points, directions, components, centre)

    documents = centre + points @ components
    expected = torch.einsum(
        "gpd,vd->vgp", torch.nn.functional.normalize(documents, dim=-1), directions
    )
    torch.testing.assert_close(found, expected)


def test_the_identification_gain_is_log_n_for_clouds_on_their_own_views_and_zero_for_alike_clouds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_flow, "KAPPAS", (1000.0,))
    directions = torch.eye(3)
    held = text_flow.HeldOut(
        torch.arange(3),
        torch.zeros(3, len(GRAM_NAMES)),
        torch.arange(3),
        torch.zeros(3, dtype=torch.long),
        directions,
    )
    bases = {
        "components": torch.eye(3).expand(len(VIEW_TYPES), -1, -1),
        "centre": torch.zeros(len(VIEW_TYPES), 3),
    }
    own = directions[:, None, None].expand(-1, len(VIEW_TYPES), 2, -1)
    alike = torch.ones(3, len(VIEW_TYPES), 2, 3)

    separate = text_flow.gains(text_flow.cloud_scores(own, held, bases), held)
    same = text_flow.gains(text_flow.cloud_scores(alike, held, bases), held)

    assert separate[0, 0].item() == pytest.approx(math.log(3))
    assert same[0, 0].item() == pytest.approx(0.0, abs=1e-4)
    assert torch.isnan(separate[0, 1:]).all()


def test_clouds_hold_each_galaxy_s_points_by_view_type_in_their_type_s_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {"POINTS": 3, "RANK": 2, "EULER": 4}.items():
        monkeypatch.setattr(text_flow, name, value)

    def network(x, t, condition, present, kind):
        return torch.stack([1000 * condition[:, 0], 1000 * kind.float()], dim=-1)

    conditions = torch.arange(4.0)[:, None].expand(-1, 5)
    present = torch.ones(4, len(GRAM_NAMES))
    scale = torch.arange(1.0, len(VIEW_TYPES) + 1)[:, None].expand(-1, 2)

    found = text_flow.clouds(network, conditions, present, scale)

    galaxy, kind = (found / scale[:, None] / 1000).round().unbind(-1)
    assert found.shape == (4, len(VIEW_TYPES), 3, 2)
    assert (galaxy == torch.arange(4.0)[:, None, None]).all()
    assert (kind == torch.arange(float(len(VIEW_TYPES)))[:, None]).all()


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
        text_flow.condition(rows, present, bases),
        text_flow.condition(changed, present, bases),
    )


def test_a_view_type_too_rare_for_its_own_basis_gets_its_mode_s() -> None:
    counts = torch.full((len(VIEW_TYPES),), 40)
    rare = VIEW_TYPES.index(("observed", "sdss_spectrum"))
    counts[rare] = 3
    view_types = torch.repeat_interleave(torch.arange(len(VIEW_TYPES)), counts)
    embeddings = torch.randn(len(view_types), GEMMA_DIM)

    bases = text_flow.fitted_bases(view_types, embeddings)

    components = bases["components"][rare]
    torch.testing.assert_close(
        components @ components.T, torch.eye(text_flow.RANK), atol=1e-4, rtol=0
    )
    torch.testing.assert_close(
        bases["mean"][rare], embeddings[view_types == rare].mean(dim=0)
    )


def test_the_log_density_of_a_linear_flow_follows_its_divergence() -> None:
    network = text_flow.Flow(text_flow.CONDITION, 1)
    rate = 0.3
    network.velocity = lambda x, context: rate * x
    network.context = lambda t, *conditions: torch.zeros(len(t), 1)
    points = torch.randn(5, text_flow.CONDITION)

    found = text_flow.log_density(network, points)

    start = points * (1 - rate / text_flow.EULER) ** text_flow.EULER
    normal = -0.5 * (start**2).sum(-1) - 0.5 * text_flow.CONDITION * math.log(
        2 * math.pi
    )
    torch.testing.assert_close(found, normal - rate * text_flow.CONDITION)


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


def test_training_stops_after_the_patience_without_gain_and_keeps_the_best_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    rng = np.random.default_rng(0)
    held = np.array([1, 5, 9])
    training = np.setdiff1d(np.arange(12), held)
    joint = rng.standard_normal((12, len(GRAM_NAMES) * 2)).astype(np.float32)
    np.save(tmp_path / "joint_embedding.npy", joint)
    folder = tmp_path / "text_views.parquet"
    folder.mkdir()
    for index in range(2):
        written_views(folder / f"part-{index:05d}.parquet", training, rng, index)
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
            path = folder / f"part-{index:05d}.parquet"
            if not path.exists():
                written_views(path, training, rng, index)

    measured = iter([0.1, 0.5, 0.505, 0.4, 0.45])

    snapshots = []

    def identification(network, held, bases, present):
        snapshots.append(deepcopy(network.state_dict()))
        kappas = len(text_flow.KAPPAS)
        found = torch.full((kappas, len(VIEW_TYPES)), next(measured, 0.0))
        return found + torch.arange(kappas)[:, None] / 1000

    for module in (text_flow, text_views, embeddings):
        monkeypatch.setattr(
            module, "artifact", lambda role: tmp_path / f"{role}.{ARTIFACTS[role]}"
        )
    monkeypatch.setattr(text_views, "training", lambda: training)
    monkeypatch.setattr(text_views, "generate_text_views", generate_text_views)
    monkeypatch.setattr(text_flow, "identification", identification)
    for name, value in {
        "RANK": 3,
        "CONDITION": 4,
        "HIDDEN": 8,
        "BATCH": 8,
        "CHUNK": len(VIEW_TYPES),
        "STEPS": 2,
        "PATIENCE": 2,
        "POINTS": 2,
        "EULER": 2,
        "KAPPAS": (1.0, 2.0, 3.0),
        "CAP": 10 * 2 * len(VIEW_TYPES),
        "VIEWS_PER_PART": 2 * len(VIEW_TYPES),
        "PARTS_PER_ROUND": 2,
    }.items():
        monkeypatch.setattr(text_flow, name, value)

    text_flow.train_text_flow()

    printed = capsys.readouterr().out
    report = json.loads(printed[printed.find("{") :])
    stored = np.load(tmp_path / "text_bases.npz")
    flow = torch.load(tmp_path / "text_flow.pt")
    assert generated == [2, 4]
    assert report["views"] == 2 * 2 * len(VIEW_TYPES)
    assert stored["kappa"] == 3.0
    assert stored["components"].shape == (len(VIEW_TYPES), 3, GEMMA_DIM)
    assert stored["projection"].shape == (len(GRAM_NAMES) * 2, 4)
    assert all(torch.equal(flow[key], snapshots[1][key]) for key in flow)
    assert not all(torch.equal(flow[key], snapshots[3][key]) for key in flow)
