Text search
===========

Text search uses a second model, `EmbeddingGemma <https://huggingface.co/google/embeddinggemma-2>`_, which places images and text in one 768-dimensional space where a picture and a description of it lie close together.

For each galaxy, alphaUniverse pairs two vectors: AION's mean embedding :math:`a_g`, and EmbeddingGemma's embedding :math:`e_g` of its Legacy Survey colour image. A small network

.. math::

   f(a) = \frac{W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2}
   {\lVert W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2 \rVert},
   \qquad W_1 \in \mathbb{R}^{2048 \times 768},

is trained on 70% of the galaxies to minimise :math:`1 - \cos(f(a_g), e_g)`, carrying AION's view of each galaxy into EmbeddingGemma's space. A query is embedded by EmbeddingGemma as :math:`t`, and galaxies are ranked by :math:`\cos(f(a_g), t)` (``app.text_search.text_search``). :doc:`/benchmarks` measures how well these rankings recover PROVABGS properties.
