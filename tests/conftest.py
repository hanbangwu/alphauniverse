"""Fixtures backed by a synthetic artifact tree.

The whole suite runs against one tree built once per session by
``scripts.fixture``. Because :func:`app.config.build_dir` reads the environment
on every call, pointing the app at the tree is just an environment variable —
but the module-level ``@cache`` decorators hold artifacts open, so they are
cleared whenever the tree changes underneath them.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

GALAXIES = 12


def _forget() -> None:
    """Drop every cached handle onto the artifact tree."""
    from app import main, search

    search.source.cache_clear()
    search.index.cache_clear()
    main.labels.cache_clear()


@pytest.fixture(scope="session")
def galaxies() -> int:
    """How many galaxies the fixture tree holds."""
    return GALAXIES


@pytest.fixture(scope="session")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A built artifact tree, with the app pointed at it."""
    from scripts.fixture import build

    os.environ["ALPHAUNIVERSE_CACHE"] = str(tmp_path_factory.mktemp("artifacts"))
    _forget()
    target = build(GALAXIES)
    _forget()
    yield target
    _forget()


@pytest.fixture(scope="session")
def client(tree: Path) -> Iterator[TestClient]:
    """An HTTP client over the app, with lifespan startup run."""
    from app.main import app

    with TestClient(app) as opened:
        yield opened
