How-to guides
=============

Every command runs from the repository root and needs `uv <https://docs.astral.sh/uv/>`_; the web app also needs `bun <https://bun.sh>`_.

Run the site locally
--------------------

Build the dev subset on Modal and download it into ``.cache``, then serve the API from it and the web app, which reaches ``http://127.0.0.1:8000`` through Vite's ``/api`` proxy:

.. code-block:: sh

   uv run python -m scripts.dev_subset   # --galaxies, default 128
   ACCESS_PASSWORD=local uv run fastapi dev server/main.py
   cd web && bun install && bun run dev

Enter the same password at ``/login``. ``MODAL_ENVIRONMENT=dev`` before the first command builds the dev subset from the ``dev`` volume. Develop against the dev subset, never measure with it. Without ``.cache/dataset`` the API reads the Hugging Face dataset (24.33 GB at the pinned revision). The first text search downloads WeMM-Embedding.

Build the dataset and the artifacts
-----------------------------------

The dataset is built outside the deployed pipeline:

.. code-block:: sh

   uv run --group dataset python -m scripts.alphauniverse_cosmos

The library builds the artifacts search reads, in order, on the machine it runs on, into ``ALPHAUNIVERSE_DATA_DIR`` (``.cache`` by default); the AION steps need a GPU:

.. code-block:: sh

   uv run --extra build python -c "from alphauniverse.build import build; build()"

The server's map artifacts are then built on Modal, from the library's artifacts on the volume; see :doc:`reference` for what each writes:

.. code-block:: sh

   uv run modal run modal_app.py::generate_properties
   uv run modal run modal_app.py::generate_maps
   uv run modal run modal_app.py::generate_projections

``benchmarks.library.predictions.build.observed_grams`` builds the observed Grams and AION prediction embeddings that two library benchmarks read:

.. code-block:: sh

   uv run modal run -m benchmarks.library.predictions.build.observed_grams

A failed job leaves its artifacts incomplete; rerun it. Text artifacts are not keyed by text model: after changing ``WEMM``, move ``text_views``, ``text_views_held_out``, ``text_regression``, ``text_bases``, ``text_vectors`` and ``text_query_mean`` aside before rerunning ``generate_text_views``, ``train_text_regression`` and ``generate_text_vectors``, or new views are mixed with old ones, and rerun ``generate_pairs``, whose ``pairs_image`` and ``pairs_spectrum`` hold the text model's embeddings that ``text_search_quality`` scores queries against.

These commands write to the volume production serves from. To build into the ``dev`` Modal environment instead, which has its own app, volume and secrets, add ``--env dev`` after ``modal run``.

Migrate artifacts to ``object_id`` keys
----------------------------------------

Artifacts built before the library split key rows by an integer ``galaxy`` and carry the Galaxy Zoo 10 and PROVABGS columns. ``scripts/migrate_object_ids.py`` rewrites them on a volume in place, without recomputing anything, and skips files already migrated. The served app cannot read the other format, so deploy right after migrating:

.. code-block:: sh

   uv run modal run --env dev scripts/migrate_object_ids.py
   uv run modal run scripts/migrate_object_ids.py

Preview a branch
----------------

Run the CI/CD workflow by hand on the branch to deploy it to the ``dev`` app:

.. code-block:: sh

   gh workflow run ci-cd.yml --ref <branch>

The branch's Vercel preview then calls that app.

Update the API schema
---------------------

After changing a route or response model in ``server/main.py`` or a module's ``routes.py``:

.. code-block:: sh

   uv run python -m scripts.openapi
   cd web && bun run check

The first regenerates ``web/openapi.json``; the second regenerates the web app's client from it and typechecks.

Run the tests and linters
-------------------------

.. code-block:: sh

   uv run --group build pytest
   uv run ruff check src server benchmarks scripts tests modal_app.py
   uv run ruff format src server benchmarks scripts tests modal_app.py
   uv run lint-imports
   cd web && bun run lint && bun run check

Tests need no Modal, GPU or network. Each run replaces ``.fixtures`` with the fixture.

Run a benchmark
---------------

.. code-block:: sh

   uv run modal run -m benchmarks.library.search.search_performance                            # --runs, default 30
   uv run modal run -m benchmarks.library.search.search_quality                                # --sample, default 1000
   uv run modal run -m benchmarks.server.backend_performance                                  # --runs, default 30
   uv run modal run -m benchmarks.server.projection.build.projection_quality
   uv run modal run -m benchmarks.library.text_search.text_search_quality                      # --half, default development, or sealed
   uv run modal run -m benchmarks.library.text_search.text_search_sheet                        # --out, default text_search_sheet.html; --earlier, an earlier sheet's .json
   uv run --group build modal run -m benchmarks.library.predictions.build.compression_quality  # --sample 1024, --queries 200, --modes every mode
   uv run modal run -m benchmarks.library.predictions.build.token_embedding_quality
   uv run --group build modal run -m benchmarks.library.predictions.build.aion_prediction_quality
   uv run modal run -m benchmarks.library.predictions.build.aion_prediction_embedding_quality
   uv run --group build modal run -m benchmarks.library.predictions.build.retrieval_quality
   uv run --group build modal run -m benchmarks.library.predictions.build.redshift_quality
   uv run --group build modal run -m benchmarks.server.projection.build.neighbor_quality

Each writes its report to ``docs/benchmarks/<script>.json``.

From GitHub, start the Benchmark workflow from the Actions tab, pick a branch, a script and optionally its flags. The report is the run's artifact; the workflow does not commit it. A job past six hours stops without a report.

Build this documentation
------------------------

.. code-block:: sh

   uv run --group docs make -C docs html

The site is written to ``docs/_build/html``. Any warning fails the build, as it does in CI. The benchmark pages read the reports relative to ``docs/``, so build from the Makefile.
