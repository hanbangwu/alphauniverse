import hashlib
from pathlib import Path

import numpy as np
import PIL.Image
import pytest

from app import template_gemma
from app.config import GEMMA_DIM, MODES, artifact
from app.search import templates


class FakeModel:
    def encode(
        self, documents: list[PIL.Image.Image] | list[str], prompt_name: str
    ) -> np.ndarray:
        return np.stack([self.vector(document) for document in documents])

    def vector(self, document: PIL.Image.Image | str) -> np.ndarray:
        content = document.encode() if isinstance(document, str) else document.tobytes()
        digest = np.frombuffer(hashlib.sha256(content).digest(), np.uint8)
        return np.resize(digest, GEMMA_DIM).astype(np.float32)


def test_each_galaxy_s_row_embeds_its_own_template(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(template_gemma, "text_model", FakeModel)
    monkeypatch.setattr(template_gemma, "BATCH", 5)

    template_gemma.generate_template_gemma()

    for mode in MODES:
        expected = FakeModel().encode(
            template_gemma.documents(mode, templates(mode)), "Document"
        )
        np.testing.assert_array_equal(
            np.load(artifact(f"template_gemma_{mode}")), expected
        )
