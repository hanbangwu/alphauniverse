How-to guides
=============

Every command runs from the repository root and needs `uv <https://docs.astral.sh/uv/>`_; the frontend also needs `bun <https://bun.sh>`_.

Run the site locally
--------------------

Serve the API from a small synthetic artifact tree in the production schemas, then the frontend, which calls ``http://127.0.0.1:8000``:

.. code-block:: sh

   uv run python -m scripts.fixture
   ALPHAUNIVERSE_CACHE=.cache/fixture uv run fastapi dev app/main.py
   cd frontend && bun install && bun run dev

The fixture's 12 galaxies are synthetic: develop against it, never measure with it. Images, spectra and the catalogue table read the Hugging Face dataset itself, which the first such request downloads (24.33 GB at the pinned revision); with ``HF_HUB_OFFLINE=1`` they fail unless it is cached. The first text search downloads EmbeddingGemma the same way.

Build the dataset and the artifacts
-----------------------------------

The dataset is built outside the deployed pipeline:

.. code-block:: sh

   uv run --group dataset python -m scripts.alphauniverse_cosmos

The artifacts are built on Modal, in this order; see :doc:`reference` for what each writes:

.. code-block:: sh

   uv run modal run modal_app.py::generate_embeddings
   uv run modal run modal_app.py::generate_predictions
   uv run modal run modal_app.py::generate_token_grams
   uv run modal run modal_app.py::generate_token_embeddings
   uv run modal run modal_app.py::generate_templates
   uv run modal run modal_app.py::generate_observed_grams
   uv run modal run modal_app.py::generate_template_embeddings
   uv run modal run modal_app.py::generate_projections
   uv run modal run modal_app.py::generate_pairs
   uv run modal run modal_app.py::generate_alignment
   uv run modal run modal_app.py::generate_aion_gemma_space

A failed job leaves its artifacts incomplete; rerun it.

These commands write to the volume production serves from. To build into the ``dev`` Modal environment instead, which has its own app, volume and secrets, add ``--env dev`` after ``modal run``.

Update the API schema
---------------------

After changing a route or response model in ``app/main.py``:

.. code-block:: sh

   uv run python -m scripts.openapi
   cd frontend && bun run check

The first regenerates ``frontend/openapi.json``; the second regenerates the frontend's client from it and typechecks.

Run the tests and linters
-------------------------

.. code-block:: sh

   uv run --group build pytest
   uv run ruff check app scripts tests modal_app.py
   uv run ruff format app scripts tests modal_app.py
   cd frontend && bun run lint && bun run check

Tests need no Modal, GPU or network.

Run a benchmark
---------------

.. code-block:: sh

   uv run modal run -m scripts.benchmarks.search_performance                 # --runs, default 30
   uv run modal run -m scripts.benchmarks.search_quality                     # --sample, default 1000
   uv run modal run -m scripts.benchmarks.backend_performance                # --runs, default 30
   uv run modal run -m scripts.benchmarks.projection_quality
   uv run modal run -m scripts.benchmarks.text_search_quality
   uv run --group build modal run -m scripts.benchmarks.compression_quality  # --sample 1024, --queries 200, --modes every mode
   uv run modal run -m scripts.benchmarks.token_embedding_quality
   uv run --group build modal run -m scripts.benchmarks.template_quality
   uv run modal run -m scripts.benchmarks.template_embedding_quality
   uv run --group build modal run -m scripts.benchmarks.retrieval_quality
   uv run --group build modal run -m scripts.benchmarks.probe_quality

Each writes its report to ``docs/benchmarks/<script>.json``.

From GitHub, start the Benchmark workflow from the Actions tab, pick a branch, a script and optionally its flags. The report appears in the run's summary and as its artifact; the workflow does not commit it. A job past six hours stops without a report.

Build this documentation
------------------------

.. code-block:: sh

   uv run --group docs make -C docs html

The site is written to ``docs/_build/html``. Any warning fails the build, as it does in CI. The benchmark pages read the reports relative to ``docs/``, so build from the Makefile.
