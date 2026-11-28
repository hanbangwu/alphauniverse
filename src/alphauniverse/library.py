from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal, get_args

import astropy.units as u
import numpy as np
from astropy.table import Table
from numpy.typing import ArrayLike

from alphauniverse import config
from alphauniverse.config import AION_SPECTRUM_GRID, REDSHIFT
from alphauniverse.encoding.store import object_ids, rows
from alphauniverse.predictions.store import aion_predictions
from alphauniverse.search import pql
from alphauniverse.text_search.text_search import (
    TextQuery,
    embed_queries,
    text_search,
)

GalaxyId = str

ImageSurvey = Literal["ls", "hsc"]
SpectrumSurvey = Literal["desi", "sdss"]
TableSurvey = Literal["ls", "hsc", "desi", "sdss"]

TokenMode = Literal["ls_image", "hsc_image", "desi_spectrum", "sdss_spectrum"]

Mode = Literal[
    "ls_image",
    "hsc_image",
    "desi_spectrum",
    "sdss_spectrum",
    "ls_table",
    "hsc_table",
    "desi_table",
    "sdss_table",
]

IMAGE_TOKEN_GRID: tuple[int, int] = (24, 24)

SPECTRUM_TOKENS: int = 272
SPECTRUM_START: u.Quantity = 3500 * u.AA
SPECTRUM_TOKEN_WIDTH: u.Quantity = 25.6 * u.AA

LS_TABLE_COLUMNS: tuple[str, ...] = (
    "EBV",
    "FLUX_G",
    "FLUX_R",
    "FLUX_I",
    "FLUX_Z",
    "FLUX_W1",
    "FLUX_W2",
    "FLUX_W3",
    "FLUX_W4",
    "SHAPE_R",
    "SHAPE_E1",
    "SHAPE_E2",
)
HSC_TABLE_COLUMNS: tuple[str, ...] = (
    "a_g",
    "a_r",
    "a_i",
    "a_z",
    "a_y",
    "g_cmodel_mag",
    "r_cmodel_mag",
    "i_cmodel_mag",
    "z_cmodel_mag",
    "y_cmodel_mag",
    "i_sdssshape_shape11",
    "i_sdssshape_shape22",
    "i_sdssshape_shape12",
)

INTERNAL_MODES: dict[str, str] = {
    "ls_image": "ls_image",
    "hsc_image": "hsc_image",
    "desi_spectrum": "desi_spectrum",
    "sdss_spectrum": "sdss_spectrum",
    "ls_table": "ls_table",
    "hsc_table": "hsc_table",
    "desi_table": REDSHIFT,
    "sdss_table": REDSHIFT,
}
TOKEN_MODES = get_args(TokenMode)
TABLE_VALUES = {slot: index for index, slot in enumerate(pql.TABLE_SLOTS)}
TABLE_SPANS = {
    "desi_table": slice(0, 1),
    "sdss_table": slice(0, 1),
    "ls_table": slice(1, 1 + len(LS_TABLE_COLUMNS)),
    "hsc_table": slice(1 + len(LS_TABLE_COLUMNS), None),
}


@dataclass(frozen=True)
class SearchByPatchResults:
    """Galaxies matching selected tokens of a query galaxy.

    Every field has one entry per galaxy: the query galaxy first, then the
    matches, best first.

    Attributes
    ----------
    galaxies : list of str
        Legacy Survey ``object_id`` of each galaxy.
    scores : numpy.ndarray
        Score the galaxies are ranked by.
    similarity : numpy.ndarray
        Similarity to the query galaxy, from 0 to 1.
    modality_scores : dict of str to numpy.ndarray
        Each selected modality's score: the sum over its selected tokens of
        the log probability that the two galaxies' predicted tokens agree.
    maps : dict of str to numpy.ndarray
        Each galaxy's agreement with the query galaxy at every token of every
        modality, relative to the query's most probable code: shape
        (galaxies, 24, 24) for images, (galaxies, 272) for spectra and
        (galaxies, columns) for scalars.
    predicted : list of list of str
        Selected modalities each galaxy lacks, matched on AION's prediction.
    positions : dict of str to numpy.ndarray
        For each ``anywhere`` modality, shape (galaxies, selected tokens): the
        token position of each galaxy that best matched each selected token.
    """

    galaxies: list[GalaxyId]
    scores: np.ndarray
    similarity: np.ndarray
    modality_scores: dict[Mode, np.ndarray]
    maps: dict[Mode, np.ndarray]
    predicted: list[list[Mode]]
    positions: dict[TokenMode, np.ndarray]


@dataclass(frozen=True)
class SearchByModeResults:
    """Galaxies matching whole modalities of a query galaxy.

    Every field has one entry per galaxy: the query galaxy first, then the
    matches, best first.

    Attributes
    ----------
    galaxies : list of str
        Legacy Survey ``object_id`` of each galaxy.
    scores : numpy.ndarray
        Score the galaxies are ranked by.
    modality_scores : dict of str to numpy.ndarray
        Each modality's contribution to ``scores``.
    maps : dict of str to numpy.ndarray
        As in :class:`SearchByPatchResults`.
    predicted : list of list of str
        Modalities each galaxy lacks, matched on AION's prediction.
    """

    galaxies: list[GalaxyId]
    scores: np.ndarray
    modality_scores: dict[Mode, np.ndarray]
    maps: dict[Mode, np.ndarray]
    predicted: list[list[Mode]]


@dataclass(frozen=True)
class SearchByTextResults:
    """Galaxies matching a text description, best first.

    Attributes
    ----------
    galaxies : list of str
        Legacy Survey ``object_id`` of each galaxy.
    scores : numpy.ndarray
        Agreement of each galaxy with the text.
    modality_scores : dict of str to numpy.ndarray
        Each modality's contribution to ``scores``.
    """

    galaxies: list[GalaxyId]
    scores: np.ndarray
    modality_scores: dict[Mode, np.ndarray]


def row(galaxy: GalaxyId) -> int:
    return int(rows([galaxy])[0])


def identified(found: np.ndarray) -> list[GalaxyId]:
    return object_ids().take(found).to_pylist()


def predict_image(galaxy: GalaxyId, survey: ImageSurvey) -> np.ndarray:
    """Return AION's prediction of a galaxy's image.

    Parameters
    ----------
    galaxy : str
        Legacy Survey ``object_id``.
    survey : {"ls", "hsc"}
        Legacy Survey (bands g, r, i, z) or HSC (bands g, r, i, z, y).

    Returns
    -------
    numpy.ndarray
        Flux of shape (bands, 96, 96) in the survey's units: the codec's
        decoding of the expected latent under AION's predicted distribution at
        every token.

    Raises
    ------
    ValueError
        If the galaxy is unknown.
    """
    return np.asarray(aion_predictions(f"{survey}_image")[row(galaxy)])


def predict_spectrum(galaxy: GalaxyId, survey: SpectrumSurvey) -> Table:
    """Return AION's prediction of a galaxy's spectrum.

    Parameters
    ----------
    galaxy : str
        Legacy Survey ``object_id``.
    survey : {"desi", "sdss"}
        DESI or SDSS.

    Returns
    -------
    astropy.table.Table
        ``wavelength`` in Å on AION's grid and ``flux``: the codec's decoding of
        the expected latent at every token. AION does not predict the overall
        flux, so the flux is normalized.

    Raises
    ------
    ValueError
        If the galaxy is unknown.
    """
    return Table(
        {
            "wavelength": AION_SPECTRUM_GRID * u.AA,
            "flux": np.asarray(aion_predictions(f"{survey}_spectrum")[row(galaxy)]),
        }
    )


def predict_tabular(galaxy: GalaxyId, survey: TableSurvey) -> Table:
    """Return AION's prediction of a galaxy's scalars from one survey.

    Parameters
    ----------
    galaxy : str
        Legacy Survey ``object_id``.
    survey : {"ls", "hsc", "desi", "sdss"}
        ``"ls"`` and ``"hsc"`` give the columns of :data:`LS_TABLE_COLUMNS`
        and :data:`HSC_TABLE_COLUMNS`. ``"desi"`` and ``"sdss"`` give the
        redshift ``Z``: AION predicts one, the same for both.

    Returns
    -------
    astropy.table.Table
        One row: the median of AION's predicted distribution of each value.

    Raises
    ------
    ValueError
        If the galaxy is unknown.
    """
    names = {"ls": LS_TABLE_COLUMNS, "hsc": HSC_TABLE_COLUMNS}.get(survey, ("Z",))
    mode = f"{survey}_table" if survey in ("ls", "hsc") else REDSHIFT
    values = np.asarray(aion_predictions(mode)[row(galaxy)])
    return Table(rows=[values.tolist()], names=names)


def slots(mask: ArrayLike, shape: tuple[int, ...], name: str) -> tuple[int, ...]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != shape:
        raise ValueError(f"{name} has shape {selected.shape}, not {shape}")
    return tuple(np.flatnonzero(selected).tolist())


def selection(
    galaxy: GalaxyId, masks: dict[str, ArrayLike | None], anywhere: Collection[str]
) -> dict:
    shapes = {
        "ls_image": IMAGE_TOKEN_GRID,
        "hsc_image": IMAGE_TOKEN_GRID,
        "desi_spectrum": (SPECTRUM_TOKENS,),
        "sdss_spectrum": (SPECTRUM_TOKENS,),
        "ls_table": (len(LS_TABLE_COLUMNS),),
        "hsc_table": (len(HSC_TABLE_COLUMNS),),
        "desi_table": (1,),
        "sdss_table": (1,),
    }
    chosen = {
        name: slots(mask, shapes[name], name)
        for name, mask in masks.items()
        if mask is not None
    }
    return {
        "galaxy": row(galaxy),
        **{name: found for name, found in chosen.items() if name in TOKEN_MODES},
        "table_values": tuple(
            sorted(
                {
                    TABLE_VALUES[(INTERNAL_MODES[name], slot)]
                    for name, found in chosen.items()
                    if name not in TOKEN_MODES
                    for slot in found
                }
            )
        ),
        "anywhere": tuple(anywhere),
    }


def selected(masks: dict[str, ArrayLike | None]) -> list[str]:
    return [
        name
        for name, mask in masks.items()
        if mask is not None and np.asarray(mask, dtype=bool).any()
    ]


def by_modality(values: dict[str, np.ndarray]) -> dict[Mode, np.ndarray]:
    found = {
        name: values[name].reshape(*values[name].shape[:-1], *IMAGE_TOKEN_GRID)
        for name in ("ls_image", "hsc_image")
    } | {name: values[name] for name in ("desi_spectrum", "sdss_spectrum")}
    return found | {
        name: values["table_values"][..., span] for name, span in TABLE_SPANS.items()
    }


def public(
    values: dict[str, np.ndarray], names: Collection[str]
) -> dict[Mode, np.ndarray]:
    return {
        name: values[INTERNAL_MODES[name]]
        for name in INTERNAL_MODES
        if name in names and INTERNAL_MODES[name] in values
    }


def lacking(predicted: list[list[str]], names: Collection[str]) -> list[list[Mode]]:
    return [
        [
            name
            for name in INTERNAL_MODES
            if name in names and pql.OBSERVED_IN[INTERNAL_MODES[name]] in missing
        ]
        for missing in predicted
    ]


def search_by_patch(
    galaxy: GalaxyId,
    *,
    ls_image: ArrayLike | None = None,
    hsc_image: ArrayLike | None = None,
    desi_spectrum: ArrayLike | None = None,
    sdss_spectrum: ArrayLike | None = None,
    ls_table: ArrayLike | None = None,
    hsc_table: ArrayLike | None = None,
    desi_table: ArrayLike | None = None,
    sdss_table: ArrayLike | None = None,
    anywhere: Collection[TokenMode] = (),
    observed_only: bool = False,
    matches: int = 32,
) -> SearchByPatchResults:
    """Find galaxies whose AION predictions agree with a galaxy's at selected tokens.

    Each selection is a boolean mask over one modality's tokens. A galaxy
    lacking a selected modality, the query galaxy included, is matched on
    AION's prediction of it.

    Parameters
    ----------
    galaxy : str
        Legacy Survey ``object_id`` of the query galaxy.
    ls_image, hsc_image : array_like of bool, optional
        Mask of shape :data:`IMAGE_TOKEN_GRID` over the image's token
        positions, indexed [row, column].
    desi_spectrum, sdss_spectrum : array_like of bool, optional
        Mask of shape (:data:`SPECTRUM_TOKENS`,). Token ``i`` covers observed
        wavelengths ``SPECTRUM_START + i * SPECTRUM_TOKEN_WIDTH`` onward.
    ls_table, hsc_table : array_like of bool, optional
        Mask of shape (12,) or (13,), in the order of
        :data:`LS_TABLE_COLUMNS` or :data:`HSC_TABLE_COLUMNS`.
    desi_table, sdss_table : array_like of bool, optional
        Mask of shape (1,) selecting the redshift token. Both select the same
        token: AION reads one redshift.
    anywhere : collection of str
        Selected image and spectrum modalities whose tokens match at any token
        position of the other galaxy; the rest match at the same position.
    observed_only : bool
        Drop galaxies lacking a selected modality.
    matches : int
        Number of matches, 1 to 128.

    Returns
    -------
    SearchByPatchResults

    Raises
    ------
    ValueError
        If the galaxy is unknown, nothing is selected, a mask has the wrong
        shape, or ``anywhere`` names an unselected modality.
    """
    masks = {
        "ls_image": ls_image,
        "hsc_image": hsc_image,
        "desi_spectrum": desi_spectrum,
        "sdss_spectrum": sdss_spectrum,
        "ls_table": ls_table,
        "hsc_table": hsc_table,
        "desi_table": desi_table,
        "sdss_table": sdss_table,
    }
    found = pql.search(
        pql.Query(
            **selection(galaxy, masks, anywhere),
            observed_only=observed_only,
            matches=matches,
        )
    )
    names = selected(masks)
    return SearchByPatchResults(
        identified(found.galaxies),
        found.scores,
        found.similarity,
        public(found.sums, names),
        by_modality(found.maps),
        lacking(found.predicted, names),
        found.positions,
    )


def search_by_mode(
    galaxy: GalaxyId,
    modes: Collection[Mode],
    *,
    anywhere: Collection[TokenMode] = (),
    observed_only: bool = False,
    matches: int = 32,
) -> SearchByModeResults:
    """Find galaxies whose AION predictions agree with a galaxy's over whole modalities.

    This approximates :func:`search_by_patch` with every token of each
    modality selected, from scores precomputed for every pair of galaxies.

    Parameters
    ----------
    galaxy : str
        Legacy Survey ``object_id`` of the query galaxy.
    modes : collection of str
        Modalities to compare, at least one. ``"desi_table"`` and
        ``"sdss_table"`` both name the redshift token.
    anywhere : collection of str
        Image and spectrum modalities in ``modes`` matched at any token
        position.
    observed_only : bool
        Drop galaxies lacking any of ``modes``.
    matches : int
        Number of matches, 1 to 128.

    Returns
    -------
    SearchByModeResults

    Raises
    ------
    ValueError
        If the galaxy is unknown, ``modes`` is empty, or ``anywhere`` names a
        modality outside ``modes``.
    """
    found = pql.mode_search(
        pql.ModeQuery(
            galaxy=row(galaxy),
            modes=frozenset(INTERNAL_MODES[name] for name in modes),
            anywhere=frozenset(anywhere),
            observed_only=observed_only,
            matches=matches,
        )
    )
    return SearchByModeResults(
        identified(found.galaxies),
        found.scores,
        public(found.contributions, modes),
        by_modality(found.maps),
        lacking(found.predicted, modes),
    )


def search_by_text(text: str, matches: int = 32) -> SearchByTextResults:
    """Find galaxies matching a free-text description.

    Loads the WeMM-Embedding text model on first use.

    Parameters
    ----------
    text : str
        Description, 1 to 500 characters.
    matches : int
        Number of matches, 1 to 128.

    Returns
    -------
    SearchByTextResults
        ``modality_scores`` gives the redshift token's contribution under both
        ``"desi_table"`` and ``"sdss_table"``.
    """
    found = text_search(TextQuery(text=text, matches=matches), embed_queries)
    contributions = dict(
        zip(config.MODES, found.contributions.numpy(force=True).T, strict=True)
    )
    return SearchByTextResults(
        identified(found.galaxies.numpy(force=True)),
        found.scores.numpy(force=True),
        public(contributions, INTERNAL_MODES),
    )


def saliency(
    galaxy: GalaxyId,
    *,
    ls_image: ArrayLike | None = None,
    hsc_image: ArrayLike | None = None,
    desi_spectrum: ArrayLike | None = None,
    sdss_spectrum: ArrayLike | None = None,
    ls_table: ArrayLike | None = None,
    hsc_table: ArrayLike | None = None,
    desi_table: ArrayLike | None = None,
    sdss_table: ArrayLike | None = None,
    anywhere: Collection[TokenMode] = (),
) -> dict[Mode, np.ndarray]:
    """Find which tokens of a galaxy go with matching a selection.

    For every token, the partial correlation over other galaxies between their
    agreement with the query galaxy at that token and their score on the
    selection, removing their overall agreement with the query galaxy.

    Parameters
    ----------
    galaxy : str
        Legacy Survey ``object_id`` of the query galaxy.
    ls_image, hsc_image, desi_spectrum, sdss_spectrum, ls_table, hsc_table, desi_table, sdss_table : array_like of bool, optional
        The selection, as in :func:`search_by_patch`.
    anywhere : collection of str
        As in :func:`search_by_patch`, over a fixed sample of 512 galaxies.

    Returns
    -------
    dict of str to numpy.ndarray
        One value per token: shape (24, 24) for images, (272,) for spectra and
        (columns,) for scalars.

    Raises
    ------
    ValueError
        As :func:`search_by_patch`.
    """
    masks = {
        "ls_image": ls_image,
        "hsc_image": hsc_image,
        "desi_spectrum": desi_spectrum,
        "sdss_spectrum": sdss_spectrum,
        "ls_table": ls_table,
        "hsc_table": hsc_table,
        "desi_table": desi_table,
        "sdss_table": sdss_table,
    }
    found = pql.saliency(pql.Selection(**selection(galaxy, masks, anywhere)))
    return by_modality(pql.by_mode(found))
