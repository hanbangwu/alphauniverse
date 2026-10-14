import numpy as np
import torch
from tqdm import tqdm

from . import embeddings, observations, pql, templates
from .config import (
    CROP_PIXELS,
    GRID,
    IMAGE_MODES,
    MODES,
    N_SPECTRUM_TOKENS,
    SPECTRUM_MODES,
    TOKEN_GRAMS,
    artifact,
    device,
)
from .grams import ROWS

BATCH = 256


def gathered(mode: str, galaxies: np.ndarray) -> list[torch.Tensor]:
    found = []
    for start in range(0, len(galaxies), BATCH):
        part = galaxies[start : start + BATCH]
        arrays = observations.measured(mode, part)
        if not found:
            found = [
                torch.empty(
                    (len(galaxies), *values.shape[1:]),
                    dtype=torch.float64,
                    device=device(),
                )
                for values in arrays
            ]
        for target, values in zip(found, arrays, strict=True):
            target[start : start + len(part)] = torch.from_numpy(values)
    return found


def same_place(
    queries: torch.Tensor,
    reference: torch.Tensor,
    weighted: torch.Tensor,
    information: torch.Tensor,
    free: bool,
) -> torch.Tensor:
    reference = reference.double()
    reference_linear = weighted @ reference
    reference_square = information @ reference**2
    found = torch.empty(
        (len(queries), len(weighted)), dtype=torch.float64, device=device()
    )
    for start in range(0, len(queries), BATCH):
        rows = slice(start, start + BATCH)
        chunk = queries[rows].double()
        linear = chunk @ weighted.T
        square = chunk**2 @ information.T
        if free:
            found[rows] = 0.5 * (
                linear**2 / square - reference_linear**2 / reference_square
            )
        else:
            found[rows] = (linear - 0.5 * square) - (
                reference_linear - 0.5 * reference_square
            )
    return found


def saturation(
    reference: torch.Tensor,
    weighted: torch.Tensor,
    information: torch.Tensor,
    squares: torch.Tensor,
    free: bool,
) -> torch.Tensor:
    reference = reference.double()
    linear, square = weighted @ reference, information @ reference**2
    if free:
        return 0.5 * (squares - linear**2 / square)
    return 0.5 * (squares - 2 * linear + square)


def slotted(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in IMAGE_MODES:
        pixels = values.reshape(len(values), -1, CROP_PIXELS, CROP_PIXELS)
        cells = torch.nn.functional.pixel_unshuffle(pixels, CROP_PIXELS // GRID)
        return cells.flatten(2).transpose(1, 2)
    return values.reshape(len(values), N_SPECTRUM_TOKENS, -1)


def anywhere(left: torch.Tensor, right: torch.Tensor, symmetric: bool) -> torch.Tensor:
    slots = left.shape[1]
    size = ROWS // slots
    found = torch.empty((len(left), len(right)), device=device())
    for first in tqdm(range(0, len(left), size), desc="any position"):
        rows = slice(first, first + size)
        query = left[rows].flatten(0, 1)
        for second in range(first if symmetric else 0, len(right), size):
            columns = slice(second, second + size)
            gallery = right[columns]
            products = (query @ gallery.flatten(0, 1).T).view(
                -1, slots, len(gallery), slots
            )
            found[rows, columns] = products.amax(dim=3).sum(dim=1, dtype=torch.float64)
            if symmetric and second > first:
                found[columns, rows] = (
                    products.amax(dim=1).sum(dim=2, dtype=torch.float64).T
                )
    return found


def data_features(
    queries: torch.Tensor,
    weighted: torch.Tensor,
    information: torch.Tensor,
    mode: str,
    kept: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    patches = slotted(queries, mode)[:, kept]
    weighted = slotted(weighted, mode)[:, kept]
    information = slotted(information, mode)[:, kept]
    energies = 0.5 * torch.where(information > 0, weighted**2 / information, 0).sum(-1)
    energies[information.sum(-1) == 0] = torch.inf
    return (
        torch.cat((patches, patches**2, torch.ones_like(patches[..., :1])), dim=-1),
        torch.cat(
            (
                weighted.float(),
                -0.5 * information.float(),
                -energies[..., None].float(),
            ),
            dim=-1,
        ),
    )


def template_features(
    scaled: torch.Tensor, mode: str, kept: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    patches = slotted(scaled, mode)[:, kept]
    norms = -0.5 * (patches**2).sum(-1, keepdim=True)
    ones = torch.ones_like(norms)
    return (
        torch.cat((patches, norms, ones), dim=-1),
        torch.cat((patches, ones, norms), dim=-1),
    )


def template_gram(scaled: torch.Tensor) -> torch.Tensor:
    values = scaled.double()
    norms = 0.5 * (values**2).sum(dim=1)
    return values @ values.T - norms[:, None] - norms[None, :]


def grams(mode: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    queries = embeddings.loaded(f"template_{mode}", torch.float32).flatten(1)
    reference = queries.median(dim=0).values
    galaxies = observations.observed(mode)
    weighted, information, squares = gathered(mode, galaxies)
    measured = (information > 0).any(dim=1)
    galaxies = galaxies[measured.cpu().numpy()]
    weighted, information, squares = (
        part[measured] for part in (weighted, information, squares)
    )
    free = mode in SPECTRUM_MODES
    matrices = {
        f"data_gram_{mode}": same_place(queries, reference, weighted, information, free)
    }
    references = {
        "reference": reference,
        "galleries": torch.from_numpy(galaxies),
        "saturation": saturation(reference, weighted, information, squares, free),
    }
    if free:
        own = queries[galaxies].double()
        amplitude = (own * weighted).sum(1) / (own**2 * information).sum(1)
        weighted *= amplitude[:, None]
        information *= amplitude[:, None] ** 2
        references["amplitudes"] = amplitude
    depth = information.median(dim=0).values.sqrt().float()
    references["depth"] = depth
    scaled = queries * depth
    if free:
        scaled /= scaled.norm(dim=1, keepdim=True)
    matrices[f"template_gram_{mode}"] = template_gram(scaled)
    if len(TOKEN_GRAMS[mode]) > 1:
        kept = slotted(depth[None], mode)[0].sum(-1) > 0
        matrices[f"data_gram_{mode}_anywhere"] = references["saturation"] + anywhere(
            *data_features(queries, weighted, information, mode, kept),
            symmetric=False,
        )
        matrices[f"template_gram_{mode}_anywhere"] = anywhere(
            *template_features(scaled, mode, kept), symmetric=True
        )
    return matrices, references


def generate_observed_grams() -> None:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    references = {}
    for mode in tqdm(MODES, desc="observed grams"):
        matrices, values = grams(mode)
        for role, matrix in matrices.items():
            np.save(artifact(role), matrix.float().cpu().numpy())
        references |= {
            f"{mode}_{part}": tensor.cpu().numpy() for part, tensor in values.items()
        }
    np.savez(artifact("observed_reference"), **references)


def predictive_widths() -> dict[str, np.ndarray]:
    with np.load(artifact("observed_reference")) as references:
        depths = {mode: references[f"{mode}_depth"] for mode in MODES}
    count = pql.predictions().num_rows
    found = {mode: np.empty(count, np.float32) for mode in MODES}
    start = 0
    for rows in tqdm(pql.predictions().to_batches(), desc="widths"):
        for mode, values in templates.widths(rows, depths).items():
            found[mode][start : start + rows.num_rows] = values
        start += rows.num_rows
    return found


def generate_template_embeddings() -> None:
    widths = predictive_widths()
    for mode, names in TOKEN_GRAMS.items():
        for name in tqdm(names, desc=mode):
            embeddings.saved(
                f"template_embedding_{name}",
                embeddings.embedding(embeddings.loaded(f"template_gram_{name}")),
                widths=widths[mode],
            )
