import importlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

GALAXIES = 12


@pytest.fixture(scope="session")
def galaxies() -> int:
    return GALAXIES


@pytest.fixture(scope="session")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from scripts.fixture import build

    os.environ["ALPHAUNIVERSE_CACHE"] = str(tmp_path_factory.mktemp("artifacts"))
    return build(GALAXIES)


@pytest.fixture(scope="session")
def client(tree: Path) -> Iterator[TestClient]:
    from app.main import app

    with TestClient(app) as opened:
        yield opened


@pytest.fixture(scope="module")
def random_weights() -> Iterator[None]:
    torch = pytest.importorskip("torch")
    encode_module = importlib.import_module("app.encode")
    codec_config = importlib.import_module("aion.codecs.config")
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
