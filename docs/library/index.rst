Using the library
=================

``alphauniverse`` is a Python library for searching galaxies through AION's predictions and reading those predictions. The website is built on it.

Install
-------

.. code-block:: sh

   pip install "alphauniverse[text]"

The ``text`` extra adds text search; ``build`` adds what building the data needs.

Data
----

The library reads its data products from the directory named by ``ALPHAUNIVERSE_DATA_DIR``, ``.cache`` by default. Build them into it with :func:`alphauniverse.build.build`, which needs a GPU, or copy a built directory there. The ``hanbangwu/alphauniverse-cosmos`` dataset downloads from Hugging Face when the directory has no ``dataset`` folder.

Galaxies are named by their Legacy Survey ``object_id``.

Example
-------

Find the 10 galaxies whose central Legacy Survey cells AION predicts most like a galaxy's, wherever they sit:

.. code-block:: python

   import numpy as np

   import alphauniverse

   core = np.zeros(alphauniverse.IMAGE_TOKEN_GRID, bool)
   core[10:14, 10:14] = True
   found = alphauniverse.search_by_patch(
       "<object_id>", ls_image=core, anywhere=["ls_image"], matches=10
   )
   found.galaxies, found.similarity

.. toctree::
   :hidden:

   reference
