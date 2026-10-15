Text search
===========

Text search ranks galaxies by how well they match a sentence. AION reads no text, so alphaUniverse uses a second model, :term:`EmbeddingGemma`, which places images and text in one 768-dimensional space where a picture and a description of it lie close together (`model card <https://huggingface.co/google/embeddinggemma-2>`_). EmbeddingGemma sees pictures, so alphaUniverse gives it two per galaxy, the colour image and a plot of the spectrum, and carries AION's view of each observation into EmbeddingGemma's space.

.. raw:: html
   :file: ../figures/text-search.svg

Each *modality*, image or spectrum, pairs two vectors per galaxy that has it. For the image, they are the mean of AION's encoded embeddings of the galaxy's Legacy Survey image tokens, :math:`a_g`, and EmbeddingGemma's embedding :math:`e_g` of its Legacy Survey colour image. For the spectrum, they are the mean of AION's encoded embeddings of its DESI spectrum tokens, and EmbeddingGemma's embedding of its DESI spectrum, smoothed as on the site and plotted in grey on a 1680 × 384 pixel figure with no title or axis labels. Each modality has its own small network

.. math::

   f(a) = \frac{W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2}
   {\lVert W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2 \rVert},
   \qquad W_1 \in \mathbb{R}^{2048 \times 768},

with the GELU activation :cite:p:`hendrycks2016gelu`, trained to minimise :math:`1 - \cos(f(a_g), e_g)`. The same 30% of all galaxies are :term:`held out` of both networks' training for the benchmark. Every galaxy gets an image row :math:`f_\text{image}(a_g)` and every galaxy with a DESI spectrum also gets a spectrum row. A query is embedded by EmbeddingGemma as :math:`t`, and all rows are ranked together by their cosine to :math:`t` (``app.text_search.text_search``), so a galaxy can appear once for each modality.

:doc:`/benchmarks` measures how well these rankings recover PROVABGS properties, against EmbeddingGemma's own embeddings and a linear map in place of :math:`f`.
