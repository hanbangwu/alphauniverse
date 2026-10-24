import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import astropy.units as u
import lsdb
import numpy as np
import pandas as pd
import pyarrow as pa
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table, hstack
from astropy.utils.data import download_file
from datasets import Dataset
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

RADIUS = 1 * u.arcsec
REPEAT = 0.1 * u.arcsec
POSITIONS = (("ra", "dec"), ("RA", "DEC"), ("_RAJ2000", "_DEJ2000"))
ZENODO = "https://zenodo.org/api/records"
FASTSPECFIT = (
    "https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/v3.0/catalogs"
)
OUTPUT = Path(".cache") / "alphauniverse-labels"


def hats(repository: str) -> pd.DataFrame:
    catalog = lsdb.open_catalog(f"hf://datasets/{repository}")
    info = catalog.hc_structure.catalog_info
    return catalog.compute().rename(
        columns={info.ra_column: "ra", info.dec_column: "dec"}
    )


def flat(table: Table) -> pd.DataFrame:
    table.remove_columns([name for name in table.colnames if table[name].ndim > 1])
    return table.to_pandas()


def irsa(table: str) -> pd.DataFrame:
    query = urlencode({"QUERY": f"SELECT * FROM {table}", "FORMAT": "votable"})
    return flat(
        Table.read(f"https://irsa.ipac.caltech.edu/TAP/sync?{query}", format="votable")
    )


def vizier(table: str) -> pd.DataFrame:
    query = urlencode(
        {
            "-source": table,
            "-out.all": "",
            "-out.add": "_RAJ,_DEJ",
            "-out.max": "unlimited",
            "-oc.form": "dec",
        }
    )
    return flat(
        Table.read(
            f"https://vizier.cds.unistra.fr/viz-bin/votable?{query}", format="votable"
        )
    )


def fastspecfit() -> pd.DataFrame:
    index = urlopen(f"{FASTSPECFIT}/").read().decode()
    frames = []
    for name in sorted(set(re.findall(r'href="(fastspec-iron-[^"]+\.fits)"', index))):
        path = download_file(f"{FASTSPECFIT}/{name}", cache=True)
        metadata = Table.read(path, hdu="METADATA")
        fastspec = Table.read(path, hdu="FASTSPEC")
        fastspec.remove_columns(list(set(fastspec.colnames) & set(metadata.colnames)))
        frames.append(flat(hstack([metadata, fastspec])))
    return pd.concat(frames, ignore_index=True)


SOURCES = {
    "gz10": lambda: hats("UniverseTBD/mmu_gz10"),
    "provabgs": lambda: hats("UniverseTBD/mmu_desi_provabgs"),
    "desi_edr_sv3": lambda: hats("UniverseTBD/mmu_desi_edr_sv3"),
    "sdss": lambda: hats("UniverseTBD/mmu_sdss_sdss"),
    "chandra": lambda: hats("UniverseTBD/mmu_chandra_spectra"),
    "galaxy_mentions": lambda: hats("astronolan/galaxy-mentions-hats"),
    "gz_decals": lambda: pd.read_parquet(
        f"{ZENODO}/4573248/files/gz_decals_volunteers_5.parquet/content"
    ),
    "fastspecfit": fastspecfit,
    "zest": lambda: irsa("cosmos_morph_zurich_1"),
    "cassata": lambda: irsa("cosmos_morph_cassata_1_1"),
    "tasca": lambda: irsa("cosmos_morph_tasca_1_1"),
    "nair_abraham": lambda: vizier("J/ApJS/186/427/table2"),
    "efigi": lambda: vizier("J/A+A/532/A74/efigi"),
}


def labels(frame: pd.DataFrame) -> pd.DataFrame:
    ra, dec = next(pair for pair in POSITIONS if set(pair) <= set(frame.columns))
    frame = frame.reset_index(drop=True).rename(columns={ra: "ra", dec: "dec"})
    schema = pa.Schema.from_pandas(frame, preserve_index=False)
    scalars = [
        field.name
        for field in schema
        if not (pa.types.is_nested(field.type) or pa.types.is_binary(field.type))
    ]
    return frame[scalars].dropna(subset=["ra", "dec"]).reset_index(drop=True)


if __name__ == "__main__":
    tables = {name: labels(read()) for name, read in SOURCES.items()}

    members = pd.concat(
        [
            pd.DataFrame(
                {
                    "source": name,
                    "row": np.arange(len(table)),
                    "ra": table["ra"].to_numpy(dtype=float),
                    "dec": table["dec"].to_numpy(dtype=float),
                }
            )
            for name, table in tables.items()
        ],
        ignore_index=True,
    )
    coords = SkyCoord(members["ra"], members["dec"], unit="deg")
    left, right, separation, _ = search_around_sky(coords, coords, RADIUS)
    source = members["source"].to_numpy()
    linked = (source[left] != source[right]) | (separation < REPEAT)
    graph = coo_array(
        (np.ones(linked.sum()), (left[linked], right[linked])),
        shape=(len(members), len(members)),
    )
    members["galaxy"] = connected_components(graph, directed=False)[1]

    centres = (
        pd.DataFrame(coords.cartesian.xyz.value.T, columns=["x", "y", "z"])
        .groupby(members["galaxy"])
        .mean()
    )
    centre = SkyCoord(
        x=centres["x"], y=centres["y"], z=centres["z"], representation_type="cartesian"
    )
    galaxies = pd.DataFrame(
        {
            "ra": centre.spherical.lon.deg,
            "dec": centre.spherical.lat.deg,
            "sources": members.drop_duplicates(["galaxy", "source"])
            .groupby("galaxy")["source"]
            .agg(list),
        },
        index=centres.index,
    )
    columns = [galaxies]
    for name, table in tables.items():
        rows = members[members["source"] == name]
        columns.append(
            table.iloc[rows["row"]]
            .set_axis(rows["galaxy"])
            .groupby(level=0)
            .agg(list)
            .add_suffix(f"-{name}")
        )
    merged = pd.concat(columns, axis=1).sort_index()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT / "data.parquet"
    merged.to_parquet(output_file, index=False)

    Dataset.from_parquet(str(output_file)).push_to_hub("hanbangwu/alphauniverse-labels")
