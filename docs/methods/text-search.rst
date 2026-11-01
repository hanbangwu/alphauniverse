Text search
===========

Text search ranks galaxies by how close a sentence is, on average, to their predicted views. AION reads no text, so alphaUniverse uses a second model, :term:`EmbeddingGemma`, which places images and text in one 768-dimensional space where a picture and a description of it lie close together (`model card <https://huggingface.co/google/embeddinggemma-2>`_). What EmbeddingGemma can read about a galaxy are its :term:`views <view>`: its colour images, plots of its spectra, and its catalogue values and redshift written out as text.

.. raw:: html
   :file: ../figures/text-search.svg

A galaxy's :term:`cloud` stands in for every view it could have. Given the galaxy's :term:`joint embedding`, a flow carries Gaussian noise to 16 points for each of 26 view types, each the predicted EmbeddingGemma embedding of one view. The view types are the observation, the :term:`template` and a sample of AION's predictions in each of the seven modes, and five combinations of modes; a galaxy unobserved in a mode still gets that mode's observed points, the observation AION predicts for it. The flow learns from views of the galaxies outside the :term:`held out` set. Text search averages eight such clouds: :math:`\bar e_{g,k}` is the mean of galaxy :math:`g`'s 128 points of view type :math:`k`, each at unit length.

A query :math:`t` is embedded by EmbeddingGemma and, less the mean embedding of 10,080 generic queries, scaled to unit length, :math:`\tilde t`. The generic queries are every combination of three framings or none, 13 adjectives or none, 15 kinds of galaxy and 11 endings or none, such as *an image of a dusty spiral galaxy with a dust lane*. Galaxies rank by the query's mean cosine over their points, every view type weighted alike:

.. math::

   s(g \mid t) = \frac{1}{26} \sum_k \bar e_{g,k} \cdot \tilde t.

A mode's :term:`contribution` to a match is its Shapley value in the galaxy's lead over the average galaxy, :math:`s(g \mid t) - \langle s \rangle`. Since :math:`s` is a sum over view types, each type's part splits equally among the modes it needs:

.. math::

   \phi_m(g \mid t) = \frac{1}{26} \sum_{k \ni m} \frac{(\bar e_{g,k} - \langle \bar e_k \rangle) \cdot \tilde t}{|M_k|},

with :math:`M_k` the modes of view type :math:`k`, one for a single mode's type, and :math:`\langle \bar e_k \rangle` its mean over all galaxies. The contributions sum to the lead.

:doc:`/benchmarks` measures how well these rankings recover measured properties, against EmbeddingGemma's own embeddings of the observed image and spectrum.
