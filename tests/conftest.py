import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def galaxies() -> int:
    from scripts.fixture import GALAXIES

    return GALAXIES


@pytest.fixture(scope="session")
def tree() -> Iterator[Path]:
    from app import config
    from scripts.fixture import GALAXIES, build

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "BUILD_DIR", Path(__file__).parent.parent / ".fixtures")
        yield build(GALAXIES)


@pytest.fixture(scope="session")
def password() -> str:
    os.environ["ACCESS_PASSWORD"] = "test-password"
    return os.environ["ACCESS_PASSWORD"]


@pytest.fixture(scope="session")
def client(tree: Path, password: str) -> Iterator[TestClient]:
    from app.main import app

    headers = {"Authorization": f"Bearer {password}"}
    with TestClient(app, headers=headers) as opened:
        yield opened


@pytest.fixture(scope="module")
def random_weights() -> Iterator[None]:
    import torch
    from aion.codecs import config as codec_config

    from app import encode as encode_module
    from app.config import device

    configs = Path(__file__).parent / "aion"

    def load(codec_class: type, repository: str, modality: type) -> object:
        torch.manual_seed(0)
        path = configs / "codecs" / modality.name / "config.json"
        return codec_class(**json.loads(path.read_text())).eval()

    torch.manual_seed(0)
    config = json.loads((configs / "config.json").read_text())
    network = encode_module.AION(config | {"encoder_depth": 1, "decoder_depth": 1})
    network = network.to(device()).eval()
    network.requires_grad_(False)
    encode_module.CodecManager._load_codec_from_hf.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        for codec_class in set(codec_config.MODALITY_CODEC_MAPPING.values()):
            patch.setattr(codec_class, "from_pretrained", classmethod(load))
        patch.setattr(encode_module, "model", lambda: network)
        yield
    encode_module.CodecManager._load_codec_from_hf.cache_clear()
