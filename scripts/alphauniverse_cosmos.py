"""
uv run --with datasets --with lsdb python scripts/alphauniverse_cosmos.py
"""

import math
import shutil
from pathlib import Path

import lsdb
from dask.distributed import Client
from datasets import Dataset

COSMOS_CENTRE_RA = 150.11916667
COSMOS_CENTRE_DEC = 2.20583333
COSMOS_WIDTH = math.sqrt(2)

COSMOS = lsdb.BoxSearch(
    ra=(COSMOS_CENTRE_RA - COSMOS_WIDTH / 2, COSMOS_CENTRE_RA + COSMOS_WIDTH / 2),
    dec=(COSMOS_CENTRE_DEC - COSMOS_WIDTH / 2, COSMOS_CENTRE_DEC + COSMOS_WIDTH / 2),
)

ANCHOR = "hugging-science/mmu_legacysurvey_dr10_south_21"


def suffix(repo: str) -> str:
    return "-" + repo.split("/")[-1]


if __name__ == "__main__":
    client = Client(n_workers=4, threads_per_worker=4, memory_limit="auto")

    matched = lsdb.open_catalog(f"hf://datasets/{ANCHOR}", search_filter=COSMOS)

    left_suffix = suffix(ANCHOR)

    for repo in (
        "UniverseTBD/mmu_sdss_sdss",
        "UniverseTBD/mmu_gz10",
        "UniverseTBD/mmu_desi_provabgs",
        "UniverseTBD/mmu_desi_edr_sv3",
        "UniverseTBD/mmu_hsc_pdr3_dud_22.5",
    ):
        right = lsdb.open_catalog(f"hf://datasets/{repo}", search_filter=COSMOS)
        right_suffix = suffix(repo)
        matched = matched.crossmatch(
            right,
            how="left",
            suffixes=(left_suffix, right_suffix),
            suffix_method="all_columns",
        )
        left_suffix = ""
        matched = matched.drop([f"ra{right_suffix}", f"dec{right_suffix}"]).rename(
            {"_dist_arcsec": f"_dist_arcsec{right_suffix}"}
        )

    output_dir = Path(".cache") / "alphauniverse-cosmos"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "data.parquet"

    matched.compute().to_parquet(output_file)

    table = Dataset.from_parquet(str(output_file))
    table.push_to_hub("hanbangwu/alphauniverse-cosmos")
