from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import PIL.Image
import pyarrow.parquet as pq
import pytest
from aion.fourm.modality_info import MODALITY_INFO
from datasets import Dataset
from matplotlib import rcParams

from alphauniverse.config import (
    IMAGE_MODES,
    MODE_SURVEYS,
    MODES,
    PLOT_INCHES,
    SPECTRUM_MODES,
    WEMM_DIM,
    galaxy_count,
)
from alphauniverse.encoding import store as encoding
from alphauniverse.encoding.store import rows
from alphauniverse.text_search.build import alignment, text_views


def test_observed_view_types_are_drawn_only_for_galaxies_observed_in_the_mode(
    tree: Path,
) -> None:
    found = text_views.weights(np.arange(galaxy_count()))

    for index, (kind, name) in enumerate(text_views.VIEW_TYPES):
        if kind == "observed":
            np.testing.assert_array_equal(
                found[:, index] > 0, encoding.observed(MODE_SURVEYS[name]), name
            )


def test_predicted_views_are_documents_of_each_mode_s_form(
    tree: Path, random_weights: None
) -> None:
    galaxies = np.array([5, 2, 5])
    rng = np.random.default_rng(0)
    with ThreadPoolExecutor(1) as pool:
        for mode in MODES:
            for kind in ("aion_prediction", "sample"):
                found = text_views.realized(mode, kind, galaxies, rng, pool)

                assert len(found) == len(galaxies)
                if mode in IMAGE_MODES:
                    assert all(
                        picture.size == (MODALITY_INFO["tok_image"]["input_size"],) * 2
                        for picture in found
                    )
                elif mode in SPECTRUM_MODES:
                    size = tuple(
                        round(rcParams["figure.dpi"] * inches) for inches in PLOT_INCHES
                    )
                    assert all(picture.size == size for picture in found)
                else:
                    lines = [line for text in found for line in text.split("\n")]
                    assert all(line.partition(": ")[2] for line in lines)


def test_documents_with_pictures_are_one_chat_message_with_the_images_before_the_texts() -> (
    None
):
    picture = PIL.Image.new("RGB", (4, 4))

    assert text_views.joined((picture, "a: 1", picture)) == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": picture},
                {"type": "image", "image": picture},
                {"type": "text", "text": "a: 1"},
            ],
        }
    ]
    assert text_views.joined(("a: 1", "b: 2")) == "a: 1\nb: 2"


class Encoder:
    def encode_document(
        self, documents: list[str], normalize_embeddings: bool
    ) -> np.ndarray:
        return np.array([[len(found)] * WEMM_DIM for found in documents], np.float32)


@pytest.fixture
def rendered(tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    found = []

    def documents(view_type, galaxies, rng, pool):
        found.append(len(galaxies))
        return [f"{view_type} {galaxy} {rng.random()}" for galaxy in galaxies]

    rows = Dataset.from_dict({"galaxy": list(range(galaxy_count()))})
    monkeypatch.setattr(alignment, "dataset", lambda: rows)
    monkeypatch.setattr(text_views, "dataset", lambda: rows)
    monkeypatch.setattr(text_views, "documents", documents)
    monkeypatch.setattr(text_views, "text_model", Encoder)
    monkeypatch.setattr(text_views, "VIEWS_PER_PART", 6)
    monkeypatch.setattr(text_views, "HELD_OUT_GALAXIES", 3)
    monkeypatch.setattr(text_views, "artifact", lambda role: tmp_path / role)
    return found


def test_text_views_resume_without_redrawing_the_parts_already_written(
    rendered: list[int], tmp_path: Path
) -> None:
    folder = tmp_path / "text_views"

    text_views.generate_text_views(2)
    written = pq.read_table(folder)
    (folder / "part-00001.parquet").unlink()
    rendered.clear()
    text_views.generate_text_views(2)

    assert sum(rendered) == 6
    assert pq.read_table(folder).equals(written)
    assert not np.isin(rows(written.column("object_id")), text_views.held_out()).any()


def test_held_out_views_are_every_view_type_each_chosen_held_out_galaxy_has(
    rendered: list[int], tmp_path: Path
) -> None:
    text_views.generate_text_views(0)

    found = pq.read_table(tmp_path / "text_views_held_out")
    owners = rows(found.column("object_id"))
    galaxies = np.unique(owners)
    available = text_views.weights(galaxies) > 0
    pairs = zip(owners.tolist(), found.column("view_type").to_pylist())
    assert len(galaxies) == 3
    assert np.isin(galaxies, text_views.held_out()).all()
    assert sorted(pairs) == [
        (galaxy, kind)
        for galaxy, row in zip(galaxies.tolist(), available)
        for kind in np.flatnonzero(row).tolist()
    ]
