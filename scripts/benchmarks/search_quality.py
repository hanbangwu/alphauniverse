import json
from pathlib import Path
from typing import Any

import modal

from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import git, memory

app = modal.App("alphauniverse-mapped-memory")
image = build_image.add_local_python_source("modal_app")

LIMIT_MIB = 4096
STEP = 2**30
REPORT = Path("docs/benchmarks/search_quality.json")


def cgroup() -> dict[str, str]:
    values = {}
    for name in ("memory.current", "memory.max", "memory.peak"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            values[name] = path.read_text().strip()
    return values


@app.function(
    image=image,
    cpu=2,
    memory=(1024, LIMIT_MIB),
    timeout=60 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def touch_mapped() -> list[dict[str, Any]]:
    import numpy as np

    from app.config import artifact

    readings = []
    touched = 0
    for role in ("encoded", "search_index", "codebook"):
        mapped = np.memmap(artifact(role), mode="r", dtype=np.uint8)
        for start in range(0, len(mapped), STEP):
            int(mapped[start : start + STEP : 4096].sum())
            touched += min(STEP, len(mapped) - start)
            reading = {
                "role": role,
                "file_gib": round(len(mapped) / 2**30, 2),
                "touched_gib": round(touched / 2**30, 2),
                **memory(),
                **cgroup(),
            }
            print(json.dumps(reading), flush=True)
            readings.append(reading)
    return readings


@app.function(image=image, cpu=2, memory=(1024, LIMIT_MIB), timeout=30 * 60)
def allocate_anonymous() -> list[dict[str, Any]]:
    import numpy as np

    readings = []
    blocks = []
    for _ in range(8):
        blocks.append(np.ones(2**29, dtype=np.uint8))
        reading = {"allocated_gib": len(blocks) / 2, **memory(), **cgroup()}
        print(json.dumps(reading), flush=True)
        readings.append(reading)
    return readings


@app.local_entrypoint()
def main() -> None:
    report: dict[str, Any] = {"git": git(), "limit_mib": LIMIT_MIB}
    for name, function in (
        ("mapped", touch_mapped),
        ("anonymous", allocate_anonymous),
    ):
        try:
            report[name] = {"completed": True, "readings": function.remote()}
        except modal.exception.Error as error:
            report[name] = {"completed": False, "error": repr(error)}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
