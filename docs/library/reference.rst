Reference
=========

Search
------

.. autofunction:: alphauniverse.search_by_patch

.. autofunction:: alphauniverse.search_by_mode

.. autofunction:: alphauniverse.search_by_text

.. autofunction:: alphauniverse.saliency

.. autoclass:: alphauniverse.SearchByPatchResults

.. autoclass:: alphauniverse.SearchByModeResults

.. autoclass:: alphauniverse.SearchByTextResults

Predictions
-----------

.. autofunction:: alphauniverse.predict_image

.. autofunction:: alphauniverse.predict_spectrum

.. autofunction:: alphauniverse.predict_tabular

Building the data
-----------------

.. autofunction:: alphauniverse.build.build

Constants
---------

``IMAGE_TOKEN_GRID``
   ``(24, 24)``: the image token positions over the 96 × 96 px cutout, 4 × 4 px each.

``SPECTRUM_TOKENS``, ``SPECTRUM_START``, ``SPECTRUM_TOKEN_WIDTH``
   272 spectrum tokens from 3500 Å, 25.6 Å each.

``LS_TABLE_COLUMNS``, ``HSC_TABLE_COLUMNS``
   The order of the Legacy Survey and HSC scalars in masks and predictions.
