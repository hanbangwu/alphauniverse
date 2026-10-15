Text search
===========

AION reads no text, so text search uses :term:`EmbeddingGemma` (`model card <https://huggingface.co/google/embeddinggemma-2>`_). It embeds each galaxy's Legacy Survey colour image and, if the galaxy has one, its DESI spectrum, smoothed as on the site and plotted in grey on a 1680 × 384 pixel figure without labels.

.. raw:: html
   :file: ../figures/text-search.svg

A query is embedded as :math:`t`, and every image and spectrum embedding is ranked by its cosine to :math:`t`, so a galaxy can appear once for each. :doc:`/benchmarks` measures how well these rankings recover PROVABGS properties.
