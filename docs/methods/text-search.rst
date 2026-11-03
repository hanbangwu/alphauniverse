Text search
===========

Text search ranks galaxies by how close a sentence is, on average, to their predicted views. AION reads no text, so alphaUniverse uses a second model, :term:`WeMM-Embedding`, which places images and text in one space where a picture and a description of it lie close together (`model card <https://huggingface.co/tencent/WeMM-Embedding-2B>`_), cut to its first 1,024 dimensions. What WeMM-Embedding can read about a galaxy are its :term:`views <view>`: its colour images, plots of its spectra, and its catalogue values and redshift written out as text.

.. raw:: html
   :file: ../figures/text-search.svg

A network predicts every view a galaxy could have. Given the galaxy's :term:`joint embedding` and one of 26 view types, it predicts :math:`\bar e_{g,k}`, the average over galaxy :math:`g`'s views of type :math:`k` of each view's WeMM-Embedding embedding less the type's mean, at unit length. The view types are the observation, the :term:`template` and a sample of AION's predictions in each of the seven modes, and five combinations of modes; a galaxy unobserved in a mode still gets a prediction of that mode's observation, the one AION predicts for it. The network learns by least squares from views of the galaxies outside the :term:`held out` set. Being an average of unit vectors, a prediction is shorter where a galaxy's views of a type disagree, and its dot product with a unit vector is the predicted mean cosine of the views with it, within the type's 32 principal components.

A query :math:`t` is embedded by WeMM-Embedding and, less the mean embedding of 10,080 generic queries, scaled to unit length, :math:`\tilde t`. The generic queries are every combination of three framings or none, 13 adjectives or none, 15 kinds of galaxy and 11 endings or none, such as *an image of a dusty spiral galaxy with a dust lane*. Galaxies rank by the query's predicted mean cosine with their views, every view type weighted alike:

.. math::

   s(g \mid t) = \frac{1}{26} \sum_k \bar e_{g,k} \cdot \tilde t.

A mode's :term:`contribution` to a match is its Shapley value in the galaxy's lead over the average galaxy, :math:`s(g \mid t) - \langle s \rangle`. Since :math:`s` is a sum over view types, each type's part splits equally among the modes it needs:

.. math::

   \phi_m(g \mid t) = \frac{1}{26} \sum_{k \ni m} \frac{(\bar e_{g,k} - \langle \bar e_k \rangle) \cdot \tilde t}{|M_k|},

with :math:`M_k` the modes of view type :math:`k`, one for a single mode's type, and :math:`\langle \bar e_k \rangle` its mean over all galaxies. The contributions sum to the lead.

:doc:`/dev/benchmarks` measures how well these rankings recover measured properties, against WeMM-Embedding's own embeddings of the observed image and spectrum.
