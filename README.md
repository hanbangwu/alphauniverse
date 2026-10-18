# alphaUniverse

The embedding-based virtual astronomical observatory.

alphaUniverse wraps [AION](https://arxiv.org/abs/2510.17960), Polymathic's foundation model for astronomy, in a web interface over the galaxies of the COSMOS field.

## Features

- **Map**: every galaxy placed by its neighbours under whole-mode search, at any position or the same place; coloured by redshift, stellar mass, star formation rate, specific star formation rate or g − r colour, measured or predicted, and filtered to a range of it; a box or lasso selects galaxies to search from or export.
- **Galaxy view**: crossmatches, and Legacy Survey and HSC images and DESI and SDSS spectra in groups that collapse.
- **AION predictions**: a switch shows what AION predicts for any galaxy in every survey, observed or not: images in place of the observed ones, spectra over them and catalogue values beside them.
- **Similarity search**: pick image regions, wavelengths and catalogue values of a galaxy in any of its surveys, with the pointer or the keyboard or by dragging the mouse across many at once; galaxies rank by predictive query likelihood over AION's predictions, at the same place or any position, or over whole modes with each mode's contribution, with maps of where they agree that a threshold can mask; the map circles the matches the range filter shows, and each can start a new search.
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
