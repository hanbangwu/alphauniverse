from typing import Annotated

from pydantic import BaseModel, Field

from alphauniverse.config import N_TABLE_VALUES, ModeName, TokenMode
from alphauniverse.search.pql import ImageTokens, Matches, SpectrumTokens


class Selection(BaseModel):
    galaxy: str
    ls_image: ImageTokens = ()
    hsc_image: ImageTokens = ()
    desi_spectrum: SpectrumTokens = ()
    sdss_spectrum: SpectrumTokens = ()
    table_values: Annotated[
        tuple[Annotated[int, Field(ge=0, lt=N_TABLE_VALUES)], ...],
        Field(max_length=N_TABLE_VALUES),
    ] = ()
    anywhere: tuple[TokenMode, ...] = ()


class Search(Selection):
    observed_only: bool = False
    matches: Matches = 32


class ModeSearch(BaseModel):
    galaxy: str
    modes: Annotated[frozenset[ModeName], Field(min_length=1)]
    anywhere: frozenset[TokenMode] = frozenset()
    observed_only: bool = False
    matches: Matches = 32
