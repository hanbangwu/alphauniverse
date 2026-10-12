Text search
===========

Text search ranks galaxies by how well they match a sentence. AION reads no text, so alphaUniverse uses a second model, :term:`EmbeddingGemma`, which places images and text in one 768-dimensional space where a picture and a description of it lie close together (`model card <https://huggingface.co/google/embeddinggemma-2>`_). EmbeddingGemma sees only a galaxy's colour image, so alphaUniverse carries AION's view of each galaxy, which draws on every observation it has, into EmbeddingGemma's space.

.. raw:: html
   :file: ../figures/text-search.svg

For each galaxy, alphaUniverse pairs two vectors: AION's :term:`mean embedding` :math:`a_g`, and EmbeddingGemma's embedding :math:`e_g` of its Legacy Survey colour image. A small network

.. math::

   f(a) = \frac{W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2}
   {\lVert W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2 \rVert},
   \qquad W_1 \in \mathbb{R}^{2048 \times 768},

with the GELU activation :cite:p:`hendrycks2016gelu`, is trained on 70% of the galaxies to minimise :math:`1 - \cos(f(a_g), e_g)`; the other 30% are :term:`held out` for the benchmark. A query is embedded by EmbeddingGemma as :math:`t`, and galaxies are ranked by :math:`\cos(f(a_g), t)` (``app.text_search.text_search``).

:doc:`/benchmarks` measures how well these rankings recover PROVABGS properties, against EmbeddingGemma's own image embeddings and a linear map in place of :math:`f`.
