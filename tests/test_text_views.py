from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app import pql, text_views
from app.config import (
    GEMMA_DIM,
    IMAGE_MODES,
    IMAGE_VOCABULARY,
    MODE_SURVEYS,
    MODES,
    N_IMAGE_TOKENS,
    PLOT_INCHES,
    REDSHIFT,
    SPECTRUM_MODES,
    TEXT_VIEWS,
    TOP_CODES,
    artifact,
    galaxy_count,
)


class Fixed:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.full(shape, self.value)


def test_draws_at_either_end_of_the_unit_interval_skip_entries_of_zero_probability() -> (
    None
):
    probabilities = np.array([[0.0, 0.5, 0.0, 0.5], [0.3, 0.7, 0.0, 0.0]], np.float32)

    for value in (0.0, np.nextafter(1.0, 0.0)):
        found = text_views.drawn(probabilities, Fixed(value))

        assert probabilities[np.arange(len(probabilities)), found].all(), value


def tail_only(rows: pa.RecordBatch, survey: str) -> pa.RecordBatch:
    for name, value in (
        ("probabilities", 0.0),
        ("query_probabilities", 0.0),
        ("tails", 1.0),
    ):
        field = rows.schema.field(f"{survey}_{name}")
        values = np.full(rows.num_rows * field.type.list_size, value)
        rows = rows.set_column(
            rows.schema.get_field_index(field.name),
            field,
            pa.FixedSizeListArray.from_arrays(
                pa.array(values, field.type.value_type), field.type.list_size
            ),
        )
    return rows


def test_image_codes_drawn_from_the_tail_are_never_a_cell_s_stored_codes(
    tree: Path,
) -> None:
    rows = text_views.rows(np.arange(galaxy_count()))
    for mode, survey in IMAGE_MODES.items():
        found = tail_only(rows, survey)
        stored, _, _ = pql.query_image_tokens(found, survey, slice(None))

        codes = text_views.sampled_image_codes(found, mode, np.random.default_rng(0))

        assert codes.shape == (galaxy_count(), N_IMAGE_TOKENS)
        assert not (codes[..., None] == stored).any()
        assert codes.min() >= 0 and codes.max() < IMAGE_VOCABULARY
        assert len(np.unique(codes)) > TOP_CODES


def test_observed_view_types_are_drawn_only_for_galaxies_observed_in_the_mode(
    tree: Path,
) -> None:
    found = text_views.weights(np.arange(galaxy_count()))

    for index, view_type in enumerate(text_views.VIEW_TYPES):
        kind, name = view_type.split(" ", 1)
        if kind == "observed":
            np.testing.assert_array_equal(
                found[:, index] > 0, pql.observed(MODE_SURVEYS[name]), view_type
            )


def test_predicted_views_are_documents_of_each_mode_s_form(
    tree: Path, random_weights: None
) -> None:
    galaxies = np.array([5, 2, 5])
    rng = np.random.default_rng(0)
    with ThreadPoolExecutor(1) as pool:
        for mode in MODES:
            for kind in ("template", "sample"):
                found = text_views.realised(mode, kind, galaxies, rng, pool)

                assert len(found) == len(galaxies)
                if mode in IMAGE_MODES:
                    assert all(picture.size == (96, 96) for picture in found)
                elif mode in SPECTRUM_MODES:
                    size = tuple(round(100 * inches) for inches in PLOT_INCHES)
                    assert all(picture.size == size for picture in found)
                else:
                    assert all(
                        line.partition(": ")[2]
                        for text in found
                        for line in text.split("\n")
                    )


def test_observed_catalogue_values_stay_with_the_galaxy_they_belong_to(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        text_views,
        "table_values",
        lambda mode, galaxies: (galaxies[:, None].astype(float), None),
    )
    galaxies = np.array([7, 2, 5])

    with ThreadPoolExecutor(1) as pool:
        found = text_views.realised(
            REDSHIFT, "observed", galaxies, np.random.default_rng(0), pool
        )

    assert found == [f"{REDSHIFT}: {galaxy}" for galaxy in galaxies]


def test_documents_with_pictures_put_one_image_placeholder_before_the_texts() -> None:
    picture = PIL.Image.new("RGB", (4, 4))

    assert text_views.joined([picture, "a: 1", picture]) == {
        "text": "<|image|>\n<|image|>\na: 1",
        "image": [picture, picture],
    }
    assert text_views.joined(["a: 1", "b: 2"]) == "a: 1\nb: 2"


class Encoder:
    def encode(self, documents: list[str], prompt_name: str) -> np.ndarray:
        return np.array([[len(found)] * GEMMA_DIM for found in documents], np.float32)


def test_text_views_resume_without_redrawing_the_parts_already_written(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered = []

    def documents(view_type, galaxies, rng, pool):
        rendered.append(len(galaxies))
        return [f"{view_type} {galaxy} {rng.random()}" for galaxy in galaxies]

    monkeypatch.setattr(text_views, "documents", documents)
    monkeypatch.setattr(text_views, "text_model", Encoder)
    monkeypatch.setattr(text_views, "PART", 6)
    folder = artifact("text_views")

    list(text_views.generate_text_views(12))
    written = pq.read_table(folder)
    (folder / "part-00001.parquet").unlink()
    rendered.clear()
    list(text_views.generate_text_views(12))

    assert sum(rendered) == 6
    assert pq.read_table(folder).equals(written)
    assert written.schema.equals(TEXT_VIEWS)
    assert not np.isin(written.column("galaxy"), text_views.held_out()).any()
