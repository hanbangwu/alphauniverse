from io import StringIO

import astropy.units as u
import numpy as np
from astropy.table import Table

from app.dataset.dataset import identity


def identities(galaxies: np.ndarray) -> Table:
    table = Table(rows=[identity(galaxy) for galaxy in galaxies.tolist()])
    table["ra"].unit = table["dec"].unit = u.deg
    return table


def written(table: Table, format: str) -> str:
    text = StringIO()
    table.write(text, format=format)
    return text.getvalue()
