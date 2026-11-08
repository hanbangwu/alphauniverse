import shutil
from collections.abc import Callable
from itertools import batched
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from aion.fourm.modality_info import MODALITY_INFO
from aion.modalities import DESISpectrum, LegacySurveyImage
from datasets import Dataset, Features, Sequence, Value

from app import config
from app.config import (
    AION_EMBEDDING_DIM,
    AION_SPECTRUM_GRID,
    DATASET_LS_SUFFIX,
    FLAG_SURVEYS,
    GRAM_NAMES,
    HSC_BANDS,
    HSC_TABLE_COLUMNS,
    IMAGE_MODES,
    KEPT,
    LS_BANDS,
    LS_TABLE_COLUMNS,
    MAPS,
    MODES,
    PREDICTIONS,
    PROBED,
    PROPERTIES,
    REDSHIFT,
    REDSHIFT_COLUMNS,
    SPECTRUM_MODES,
    SPECTRUM_TOKEN_RANK,
    STORE_COLUMNS,
    TABLE_MODES,
    TABLE_VALUE_COLUMNS,
    TEXT_VIEWS,
    TOKEN_EMBEDDING_RANK,
    TOKEN_SURVEYS,
    TOP_CODES,
    VIEW_TYPES,
    WEMM_DIM,
    artifact,
    dataset_dir,
    galaxy_count,
    points,
    store_schema,
    store_writer,
)
from app.dataset import dataset, table_columns
from app.embeddings import generate_token_search
from app.pql import (
    agreements,
    basis,
    basket_sum,
    observed,
    prediction_basis,
    prediction_batch,
    predictions,
    sampled,
    save_prediction_basis,
    stored,
)
from app.search import aion_predictions, source, tokens
from app.text_search import text_vectors

N_MORPHOLOGIES = 10
TOKENS: dict[str, int] = {
    "ls": LegacySurveyImage.num_tokens + 12,
    "hsc": LegacySurveyImage.num_tokens + 13,
    "desi": 273,
    "sdss": 273,
    REDSHIFT: 1,
}

STRIDE: dict[str, int] = {
    "ls": 1,
    "hsc": 2,
    "desi": 3,
    "sdss": 4,
    REDSHIFT: 6,
}

GALAXIES = 12
SEED = 42
CLUSTERS = 64
NOISE = 0.05

PREDICTION_BATCH = 5

IMAGE_UNITS: dict[str, float] = {"ls": 1.0, "hsc": 10**1.8}


def covered(survey: str, galaxy: int) -> bool:
    return galaxy % STRIDE[survey] == 0


def _cells(
    rng: np.random.Generator, centres: np.ndarray, galaxies: int
) -> tuple[dict[str, list[np.ndarray | None]], dict[str, list[np.ndarray | None]]]:
    embeddings: dict[str, list[np.ndarray | None]] = {survey: [] for survey in TOKENS}
    token_cells: dict[str, list[np.ndarray | None]] = {survey: [] for survey in TOKENS}

    for galaxy in range(galaxies):
        for survey, count in TOKENS.items():
            if not covered(survey, galaxy):
                embeddings[survey].append(None)
                token_cells[survey].append(None)
                continue
            assigned = rng.integers(len(centres), size=count)
            rows = centres[assigned] + NOISE * rng.standard_normal(
                (count, AION_EMBEDDING_DIM)
            )
            embeddings[survey].append(rows.astype(np.float16))
            token_cells[survey].append(assigned.astype(np.uint32))

    return embeddings, token_cells


def _store(
    role: str,
    cells: dict[str, list[np.ndarray | None]],
    galaxies: int,
    flags: dict[str, np.ndarray],
) -> None:
    with store_writer(role) as writer:
        writer.write_table(
            pa.table(
                {
                    "galaxy": np.arange(galaxies),
                    **{
                        survey: [
                            None if cell is None else list(cell)
                            for cell in cells[survey]
                        ]
                        for survey in STORE_COLUMNS
                    },
                    **flags,
                },
                schema=store_schema(role),
            )
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _predictions(rng: np.random.Generator, galaxies: int) -> None:
    ones = np.ones((MODALITY_INFO["tok_spectrum_desi"]["vocab_size"], 1))
    orthogonal, _ = np.linalg.qr(
        np.hstack(
            (
                ones,
                rng.standard_normal(
                    (
                        MODALITY_INFO["tok_spectrum_desi"]["vocab_size"],
                        SPECTRUM_TOKEN_RANK,
                    )
                ),
            )
        )
    )
    save_prediction_basis(
        {
            survey: (
                np.full(
                    MODALITY_INFO["tok_spectrum_desi"]["vocab_size"],
                    1 / MODALITY_INFO["tok_spectrum_desi"]["vocab_size"],
                ),
                orthogonal[:, 1:].T,
            )
            for survey in SPECTRUM_MODES.values()
        }
    )
    records = []
    for galaxy in range(galaxies):
        values = {
            "galaxy": np.asarray([galaxy], dtype=np.int32),
            REDSHIFT: _softmax(
                3 * rng.standard_normal(MODALITY_INFO["tok_z"]["vocab_size"])
            ).astype(np.float32),
        }
        for survey, table_values in (
            ("ls", LS_TABLE_COLUMNS),
            ("hsc", HSC_TABLE_COLUMNS),
        ):
            mass = rng.dirichlet(
                np.full(TOP_CODES + 1, 0.3), size=LegacySurveyImage.num_tokens
            )
            codes = (
                rng.random(
                    (
                        LegacySurveyImage.num_tokens,
                        MODALITY_INFO["tok_image"]["vocab_size"],
                    )
                )
                .argpartition(TOP_CODES, axis=1)[:, :TOP_CODES]
                .astype(np.uint16)
            )
            values[f"{survey}_codes"] = codes[:, :KEPT].reshape(-1)
            values[f"{survey}_query_codes"] = codes[:, KEPT:].reshape(-1)
            probabilities = -np.sort(-mass[:, :TOP_CODES], axis=1)
            values[f"{survey}_probabilities"] = (
                probabilities[:, :KEPT].astype(np.float32).reshape(-1)
            )
            values[f"{survey}_query_probabilities"] = (
                probabilities[:, KEPT:].astype(np.float16).reshape(-1)
            )
            values[f"{survey}_rest"] = (
                probabilities[:, KEPT:].sum(axis=1) + mass[:, -1]
            ).astype(np.float32)
            values[f"{survey}_tails"] = mass[:, -1].astype(np.float32)
            values[f"{survey}_table_values"] = (
                _softmax(
                    3
                    * rng.standard_normal(
                        (
                            len(table_values),
                            MODALITY_INFO["tok_spectrum_desi"]["vocab_size"],
                        )
                    )
                )
                .astype(np.float32)
                .reshape(-1)
            )
        for survey in SPECTRUM_MODES.values():
            steps = rng.uniform(1e-6, 3e-6, DESISpectrum.num_tokens - 1).astype(
                np.float32
            )
            values[f"{survey}_coefficients"] = rng.integers(
                256,
                size=(DESISpectrum.num_tokens - 1) * SPECTRUM_TOKEN_RANK,
                dtype=np.uint8,
            )
            values[f"{survey}_offsets"] = -128 * steps
            values[f"{survey}_steps"] = steps
        records.append(values)
    with pa.ipc.new_file(artifact("probabilities"), PREDICTIONS) as writer:
        for chunk in batched(records, PREDICTION_BATCH):
            writer.write_batch(prediction_batch(list(chunk)))


def _aion_predictions(rng: np.random.Generator, galaxies: int) -> dict[str, np.ndarray]:
    centre = (MODALITY_INFO["tok_image"]["input_size"] - 1) / 2
    rows, columns = np.indices(
        (
            MODALITY_INFO["tok_image"]["input_size"],
            MODALITY_INFO["tok_image"]["input_size"],
        )
    )
    radius = np.hypot(rows - centre, columns - centre)
    profile = np.exp(-0.5 * (radius / rng.uniform(2, 8, (galaxies, 1, 1, 1))) ** 2)
    made = {}
    for mode, survey in IMAGE_MODES.items():
        bands = len(HSC_BANDS if survey == "hsc" else LS_BANDS)
        peaks = rng.uniform(0.02, 0.5, (galaxies, bands, 1, 1))
        noise = 0.003 * rng.standard_normal((galaxies, bands, *radius.shape))
        made[mode] = IMAGE_UNITS[survey] * (peaks * profile + noise)
    for mode in SPECTRUM_MODES:
        tilt = rng.uniform(-0.3, 0.3, (galaxies, 1))
        centres = rng.uniform(4000, 9500, (galaxies, 1))
        line = rng.uniform(0, 2, (galaxies, 1))
        made[mode] = (
            1
            + tilt * (AION_SPECTRUM_GRID / AION_SPECTRUM_GRID.mean() - 1)
            + line * np.exp(-0.5 * ((AION_SPECTRUM_GRID - centres) / 4) ** 2)
            + 0.02 * rng.standard_normal((galaxies, len(AION_SPECTRUM_GRID)))
        )
    for mode, (_, count, _) in TABLE_MODES.items():
        made[mode] = rng.lognormal(size=(galaxies, count))
    for mode, values in made.items():
        np.save(artifact(f"aion_prediction_{mode}"), values.astype(np.float32))
    return made


def _observed(
    survey: str, galaxies: int, cell: Callable[[int], dict]
) -> list[dict | None]:
    return [
        cell(galaxy) if covered(survey, galaxy) else None for galaxy in range(galaxies)
    ]


def _dataset(
    rng: np.random.Generator,
    galaxies: int,
    made: dict[str, np.ndarray],
    category: pa.Array,
    provabgs: np.ndarray,
) -> None:
    spectrum_feature = {
        "lambda": Sequence(Value("float32")),
        "flux": Sequence(Value("float32")),
        "ivar": Sequence(Value("float32")),
        "mask": Sequence(Value("bool")),
    }
    ls_bands = [band.lower() for band in LS_BANDS]
    hsc_bands = [band.lower() for band in HSC_BANDS]
    columns = {
        f"object_id{DATASET_LS_SUFFIX}": [
            f"fixture-{galaxy}" for galaxy in range(galaxies)
        ],
        f"ra{DATASET_LS_SUFFIX}": rng.uniform(149.5, 150.5, galaxies),
        f"dec{DATASET_LS_SUFFIX}": rng.uniform(1.5, 2.5, galaxies),
        TOKEN_SURVEYS["ls"]: [
            {"band": ls_bands, "flux": flux} for flux in made["ls_image"]
        ],
        TOKEN_SURVEYS["hsc"]: _observed(
            "hsc",
            galaxies,
            lambda galaxy: {"band": hsc_bands, "flux": made["hsc_image"][galaxy]},
        ),
        **{
            TOKEN_SURVEYS[survey]: _observed(
                survey,
                galaxies,
                lambda galaxy, survey=survey: {
                    "lambda": AION_SPECTRUM_GRID,
                    "flux": rng.uniform(0.5, 5) * made[f"{survey}_spectrum"][galaxy],
                    "ivar": np.ones(len(AION_SPECTRUM_GRID)),
                    "mask": np.zeros(len(AION_SPECTRUM_GRID), dtype=bool),
                },
            )
            for survey in SPECTRUM_MODES.values()
        },
        **{column: rng.lognormal(size=galaxies) for column in LS_TABLE_COLUMNS},
        **{
            column: [
                value if covered("hsc", galaxy) else None
                for galaxy, value in enumerate(rng.lognormal(size=galaxies))
            ]
            for column in HSC_TABLE_COLUMNS
        },
        FLAG_SURVEYS["gz10"]: category.to_pylist(),
        FLAG_SURVEYS["provabgs"]: [
            value if measured else None
            for value, measured in zip(rng.uniform(8, 12, galaxies), provabgs)
        ],
    }
    for survey, (value, error, warning) in REDSHIFT_COLUMNS.items():
        held = [covered(survey, galaxy) for galaxy in range(galaxies)]
        for column, values in (
            (value, rng.uniform(0, 1, galaxies)),
            (error, rng.uniform(0, 1e-3, galaxies)),
            (warning, np.zeros(galaxies, dtype=int)),
        ):
            columns[column] = [
                found if has else None for found, has in zip(values, held)
            ]
    Dataset.from_dict(
        columns,
        features=Features(
            {
                f"object_id{DATASET_LS_SUFFIX}": Value("string"),
                f"ra{DATASET_LS_SUFFIX}": Value("float64"),
                f"dec{DATASET_LS_SUFFIX}": Value("float64"),
                **{
                    TOKEN_SURVEYS[survey]: {
                        "band": Sequence(Value("string")),
                        "flux": Sequence(Sequence(Sequence(Value("float32")))),
                    }
                    for survey in IMAGE_MODES.values()
                },
                **{
                    TOKEN_SURVEYS[survey]: spectrum_feature
                    for survey in SPECTRUM_MODES.values()
                },
                **{column: Value("float32") for column in TABLE_VALUE_COLUMNS},
                FLAG_SURVEYS["gz10"]: Value("int64"),
                FLAG_SURVEYS["provabgs"]: Value("float32"),
                **{
                    column: Value("int64" if column == warning else "float32")
                    for value, error, warning in REDSHIFT_COLUMNS.values()
                    for column in (value, error, warning)
                },
            }
        ),
    ).save_to_disk(dataset_dir())


def _text_views(rng: np.random.Generator, galaxies: int) -> None:
    vectors = rng.standard_normal((galaxies, WEMM_DIM))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    artifact("text_views").mkdir()
    pq.write_table(
        pa.table(
            {
                "galaxy": np.arange(galaxies),
                "view_type": rng.integers(len(VIEW_TYPES), size=galaxies),
                "round": np.zeros(galaxies, int),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    vectors.astype(np.float16).ravel(), WEMM_DIM
                ),
            },
            schema=TEXT_VIEWS,
        ),
        artifact("text_views") / "part-00000.parquet",
    )


def _text_vectors(rng: np.random.Generator, galaxies: int) -> None:
    np.save(
        artifact("text_vectors"),
        rng.standard_normal((galaxies, len(MODES), WEMM_DIM)).astype(np.float32) / 64,
    )
    np.save(
        artifact("text_query_mean"), rng.standard_normal(WEMM_DIM).astype(np.float32)
    )


def _project(basis: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True) @ basis


def _properties(
    rng: np.random.Generator, galaxies: int, provabgs: np.ndarray
) -> pa.Table:
    measured = {f"{label}_measured": provabgs for label in PROBED} | {
        f"{REDSHIFT}_measured": np.array(
            [covered(REDSHIFT, galaxy) for galaxy in range(galaxies)]
        )
    }
    return pa.table(
        {"galaxy": np.arange(galaxies, dtype=np.int32)}
        | {
            name: pa.array(
                rng.random(galaxies, np.float32),
                mask=~measured.get(name, np.ones(galaxies, dtype=bool)),
            )
            for name in PROPERTIES.names[1:]
        },
        schema=PROPERTIES,
    )


def _token_embeddings(rng: np.random.Generator, galaxies: int) -> None:
    for name in GRAM_NAMES:
        np.savez(
            artifact(f"token_embedding_{name}"),
            queries=rng.standard_normal((galaxies, TOKEN_EMBEDDING_RANK), np.float32),
            galleries=rng.standard_normal((galaxies, TOKEN_EMBEDDING_RANK), np.float32),
            offsets=rng.standard_normal(galaxies, np.float32),
            self_scores=rng.standard_normal(galaxies, np.float32),
            singular_values=np.sort(rng.random(galaxies))[::-1],
        )


def forget() -> None:
    for cached in (
        galaxy_count,
        source,
        tokens,
        aion_predictions,
        text_vectors,
        predictions,
        prediction_basis,
        basis,
        observed,
        stored,
        agreements,
        sampled,
        basket_sum,
        dataset,
        table_columns,
    ):
        cached.cache_clear()


def build(galaxies: int) -> Path:
    target = config.BUILD_DIR
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)

    rng = np.random.default_rng(SEED)
    centres = rng.standard_normal((CLUSTERS, AION_EMBEDDING_DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    embeddings, token_cells = _cells(rng, centres, galaxies)

    rows = np.arange(galaxies)
    labelled = rows % 10 != 0
    flags = {"gz10": labelled, "provabgs": rows % 3 != 0}

    _store("encoded", embeddings, galaxies, flags)
    _store("tokens", token_cells, galaxies, flags)

    category = pa.array(rng.integers(N_MORPHOLOGIES, size=galaxies), mask=~labelled)

    anchors = np.stack(
        [np.asarray(cell, dtype=np.float32) for cell in embeddings["ls"]]
    )
    basis = rng.standard_normal((AION_EMBEDDING_DIM, 2)).astype(np.float32)

    pq.write_table(
        points(rows, _project(basis, anchors.mean(axis=1)), category),
        artifact("mean_points"),
        compression="zstd",
    )

    for name in MAPS:
        pq.write_table(
            points(
                rows, rng.standard_normal((galaxies, 2)).astype(np.float32), category
            ),
            artifact(f"{name}_points"),
            compression="zstd",
        )

    owner = np.repeat(rows, anchors.shape[1])
    pq.write_table(
        points(
            owner,
            _project(basis, anchors.reshape(-1, AION_EMBEDDING_DIM)),
            category.take(owner),
        ),
        artifact("full_points"),
        compression="zstd",
    )

    _predictions(rng, galaxies)
    made = _aion_predictions(rng, galaxies)
    _dataset(rng, galaxies, made, category, flags["provabgs"])
    pq.write_table(
        _properties(rng, galaxies, flags["provabgs"]), artifact("properties")
    )
    _token_embeddings(rng, galaxies)
    generate_token_search()
    _text_views(rng, galaxies)
    _text_vectors(rng, galaxies)

    forget()
    return target
