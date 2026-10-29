Text search
===========

Text search ranks galaxies by how likely a sentence is to describe them. AION reads no text, so alphaUniverse uses a second model, :term:`EmbeddingGemma`, which places images and text in one 768-dimensional space where a picture and a description of it lie close together (`model card <https://huggingface.co/google/embeddinggemma-2>`_). What EmbeddingGemma can read about a galaxy are its :term:`views <view>`: its colour images, plots of its spectra, and its catalogue values and redshift written out as text.

.. raw:: html
   :file: ../figures/text-search.svg

A galaxy's :term:`cloud` stands in for every view it could have. Given the galaxy's :term:`joint embedding`, a flow carries Gaussian noise to 16 points for each of 26 view types, 416 in all, each the predicted EmbeddingGemma embedding of one view. The view types are the observation, the :term:`template` and a sample of AION's predictions in each of the seven modes, and five combinations of modes; a galaxy unobserved in a mode still gets that mode's observed points, the observation AION predicts for it. The flow learns from views of the galaxies outside the :term:`held out` set.

A query :math:`t` is embedded by EmbeddingGemma and, less the mean embedding of 10,080 generic queries, scaled to unit length, :math:`\tilde t`. A galaxy :math:`g` with cloud :math:`C_g` scores

.. math::

   s(g \mid t) = \log \frac{1}{|C_g|} \sum_{e \in C_g} \exp\left(\kappa \cos(\tilde t, e)\right),

the log of the query's mean likelihood over the cloud, with :term:`κ` the value that best identifies held-out galaxies from their own views. Galaxies rank by :math:`s`, and :math:`p(g \mid t)` is its softmax over all galaxies. A mode's :term:`share` of a match is the part of the sum from that mode's points, and the five combination types share as one. Searching only some modes sums over their view types alone, a combination counting when all its modes are chosen, with :math:`|C_g|` their number of points.

:doc:`/benchmarks` measures how well these rankings recover measured properties, against EmbeddingGemma's own embeddings of the observed image and spectrum, and the maps from AION into EmbeddingGemma's space that text search used before.
