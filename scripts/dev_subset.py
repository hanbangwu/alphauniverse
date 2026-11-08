import argparse
import shutil
from pathlib import Path

import modal
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from modal.volume import FileEntryType

from app.config import ARTIFACTS, BUILD_DIR, MODES, SEED, artifact
from app.dataset import dataset
from modal_app import CACHE_PATH, build_image, cache_volume

FOLDER = ".dev_subset"
GALAXIES = 128
COPIED = {
    "probability_basis",
    "text_bases",
    "text_query_mean",
    "parametric_umap",
    "text_regression",
}
SHARED_PARTS = {"singular_values"}

app = modal.App("alphauniverse-dev-subset")
image = build_image.add_local_python_source("modal_app")


def table(role: str, rows: np.ndarray, target: Path) -> None:
    found = ds.dataset(artifact(role), format=ARTIFACTS[role])
    kept = found.to_table(filter=pc.field("galaxy").isin(rows))
    position = kept.schema.get_field_index("galaxy")
    kept = kept.set_column(
        position,
        kept.schema.field(position),
        pa.array(np.searchsorted(rows, kept["galaxy"].to_numpy()), pa.int32()),
    )
    if ARTIFACTS[role] == "arrow":
        with pa.ipc.new_file(target, found.schema) as writer:
            writer.write_table(kept)
    elif artifact(role).is_dir():
        target.mkdir()
        pq.write_table(kept, target / "part-00000.parquet")
    else:
        pq.write_table(kept, target)


def references(rows: np.ndarray, target: Path) -> dict[str, np.ndarray]:
    with np.load(artifact("observed_reference")) as found:
        values = dict(found)
    columns = {}
    for mode in MODES:
        galleries = values[f"{mode}_galleries"]
        kept = np.isin(galleries, rows)
        columns[mode] = np.flatnonzero(kept)
        values[f"{mode}_galleries"] = np.searchsorted(rows, galleries[kept])
        for part in ("saturation", "amplitudes"):
            if f"{mode}_{part}" in values:
                values[f"{mode}_{part}"] = values[f"{mode}_{part}"][kept]
    np.savez(target, **values)
    return columns


def matrix(role: str, rows: np.ndarray, columns: dict[str, np.ndarray]) -> np.ndarray:
    found = np.load(artifact(role), mmap_mode="r")
    if role.startswith("data_gram_"):
        mode = role.removeprefix("data_gram_").removesuffix("_anywhere")
        return found[np.ix_(rows, columns[mode])]
    if role.startswith(("token_gram_", "aion_prediction_gram_")):
        return found[np.ix_(rows, rows)]
    return found[rows]


@app.function(image=image, timeout=60 * 60, volumes={CACHE_PATH: cache_volume})
def cut(galaxies: int) -> None:
    target = Path(CACHE_PATH, FOLDER)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir()
    everyone = dataset()
    rows = np.sort(
        np.random.default_rng(SEED).choice(everyone.num_rows, galaxies, replace=False)
    )
    everyone.select(rows).save_to_disk(target / "dataset")
    columns = references(rows, target / artifact("observed_reference").name)
    for role, kind in ARTIFACTS.items():
        source, written = artifact(role), target / artifact(role).name
        if role in COPIED:
            shutil.copy(source, written)
        elif kind in ("arrow", "parquet"):
            table(role, rows, written)
        elif kind == "npy":
            np.save(written, matrix(role, rows, columns))
        elif role != "observed_reference":
            with np.load(source) as found:
                np.savez(
                    written,
                    **{
                        part: values if part in SHARED_PARTS else values[rows]
                        for part, values in found.items()
                    },
                )
    cache_volume.commit()


def download() -> None:
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    for entry in cache_volume.listdir(FOLDER, recursive=True):
        if entry.type == FileEntryType.FILE:
            path = BUILD_DIR / Path(entry.path).relative_to(FOLDER)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as file:
                cache_volume.read_file_into_fileobj(entry.path, file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxies", type=int, default=GALAXIES)
    galaxies = parser.parse_args().galaxies
    with modal.enable_output(), app.run():
        cut.remote(galaxies)
    download()
    total = sum(path.stat().st_size for path in BUILD_DIR.rglob("*") if path.is_file())
    print(f"{galaxies} galaxies -> {BUILD_DIR} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
