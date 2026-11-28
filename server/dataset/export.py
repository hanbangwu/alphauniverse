from datetime import UTC, datetime
from io import StringIO

import astropy.units as u
import numpy as np
from astropy.table import Table

from server.dataset.dataset import identity


def identities(galaxies: np.ndarray) -> Table:
    table = Table(rows=[identity(galaxy) for galaxy in galaxies.tolist()])
    table["ra"].unit = table["dec"].unit = u.deg
    return table


def written(table: Table) -> str:
    table.meta = {
        "created": datetime.now(UTC).isoformat(timespec="seconds")
    } | table.meta
    text = StringIO()
    table.write(text, format="ascii.ecsv")
    return text.getvalue()
