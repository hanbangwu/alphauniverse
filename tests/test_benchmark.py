"""Properties of the benchmark's report writing that a long run depends on."""

import json
from pathlib import Path

import pytest

from scripts.benchmark import write_report


def test_a_finished_write_leaves_no_staging_file(tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    write_report({"stages": 6}, out)

    assert list(tmp_path.iterdir()) == [out]


def test_a_write_that_dies_before_the_replace_keeps_the_last_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "bench.json"
    write_report({"stages": 6}, out)

    def interrupted(self: Path, target: Path) -> None:
        raise OSError("killed between the write and the replace")

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError):
        write_report({"stages": 1}, out)

    assert json.loads(out.read_text()) == {"stages": 6}
