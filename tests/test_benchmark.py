"""Properties of the benchmark's report writing that a long run depends on."""

import json
import os
import sys
from pathlib import Path

import pytest

from scripts import benchmark
from scripts.benchmark import write_report

STAGES = (
    "measure_sizes",
    "measure_load",
    "measure_latency",
    "measure_phases",
    "measure_recall",
)


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


def test_a_failing_stage_keeps_the_stages_before_it(
    tree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "bench.json"
    for stage in STAGES:
        monkeypatch.setattr(benchmark, stage, lambda *_: {"measured": True})

    def missing_artifact(*_: object) -> dict[str, object]:
        raise FileNotFoundError("cutouts.parquet")

    monkeypatch.setattr(benchmark, "measure_endpoints", missing_artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--tree", os.environ["ALPHAUNIVERSE_CACHE"], "--out", str(out)],
    )

    with pytest.raises(FileNotFoundError):
        benchmark.main()

    report = json.loads(out.read_text())
    assert set(report) > {"sizes", "index_load", "search", "phases", "recall"}
    assert "endpoints" not in report
