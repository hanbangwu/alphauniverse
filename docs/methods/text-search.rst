Text search
===========

AION reads no text, so text search uses a second model, :term:`EmbeddingGemma` (`model card <https://huggingface.co/google/embeddinggemma-2>`_), which places pictures and sentences in one 768-dimensional space where a picture and its description lie close together. alphaUniverse gives it two pictures per galaxy, the colour image and a plot of the spectrum, and carries AION's view of each into EmbeddingGemma's space.

.. raw:: html
   :file: ../figures/text-search.svg

For each *modality*, image or spectrum, every galaxy that has it gets a pair of vectors: :math:`a_g`, the mean of AION's embeddings of its Legacy Survey image tokens or DESI spectrum tokens, and :math:`e_g`, EmbeddingGemma's embedding of its Legacy Survey colour image or of its DESI spectrum, smoothed as on the site and plotted without labels. A small network per modality,

.. math::

   f(a) = \frac{W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2}
   {\lVert W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2 \rVert},
   \qquad W_1 \in \mathbb{R}^{2048 \times 768},

with the GELU activation :cite:p:`hendrycks2016gelu`, is trained to minimise :math:`1 - \cos(f(a_g), e_g)` on 70% of the galaxies; the other 30% are :term:`held out` for the benchmark. A sentence is embedded by EmbeddingGemma as :math:`t`, and every galaxy's image row :math:`f(a_g)`, and spectrum row where it has a DESI spectrum, are ranked together by their cosine to :math:`t`.

:doc:`/dev/benchmarks` measures how well these rankings recover PROVABGS properties.
