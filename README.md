# alphaUniverse

The embedding-based virtual astronomical observatory.

alphaUniverse wraps [AION](https://arxiv.org/abs/2510.17960), Polymathic's foundation model for astronomy, in a web interface over the galaxies of the COSMOS field.

## Features

- **Embedding map**: every galaxy on a parametric UMAP projection of AION's embeddings, one point per galaxy or per token.
- **Galaxy view**: Legacy Survey image, DESI spectrum, catalogue values and morphology.
- **Similarity search**: pick image regions, wavelengths and catalogue values of a galaxy; galaxies rank by predictive query likelihood over AION's predictions, at the same place or any position, with maps of where they agree.
- **Text search**: describe a galaxy in English; galaxies rank through EmbeddingGemma.

## Running locally

Requires [uv](https://docs.astral.sh/uv/) and [bun](https://bun.sh).

```sh
uv run python -m scripts.fixture                                   # synthetic artifacts in .cache/fixture
ALPHAUNIVERSE_CACHE=.cache/fixture uv run fastapi dev app/main.py  # API on 127.0.0.1:8000
cd frontend && bun install && bun run dev                          # frontend
```

## Documentation

How it works, benchmarks, commands and reference are in `docs/`, a Sphinx site:

```sh
uv run --group docs make -C docs html  # writes docs/_build/html
```
