The map
=======

The map places each embedding in two dimensions so that embeddings that are neighbours in 768 dimensions stay neighbours on the screen. **Mean** shows one point per galaxy, its :term:`mean embedding`; **full** shows one point per token embedding, every mode in the same space.

Only closeness on the map carries meaning. The axes have no units, and the distance between two clusters far apart says little about how different they are; a projection to two dimensions cannot keep every distance, and this one is built to keep neighbours :cite:p:`mcinnes2018umap`. :doc:`/benchmarks` measures how many neighbours it keeps.

.. raw:: html
   :file: ../figures/map.svg

How it is built
---------------

The projection is a :term:`parametric UMAP` :cite:p:`sainburg2021parametric`: a small network :math:`f_\theta: \mathbb{R}^{768} \to \mathbb{R}^2`, with layers of 256 and 128 units, applied to each embedding scaled to unit length. Because one network places every point, the token embeddings and the galaxy means share the same axes, although the means were not in its training sample.

The network is trained on 70% of 5,000,000 token embeddings drawn at random; the other 30% only monitor training. UMAP first links each embedding to its 15 nearest neighbours by cosine distance and weights each link, :math:`w_{ij}`. On the map, two points at distance :math:`d` count as neighbours with probability

.. math::

   q(d) = \frac{1}{1 + a\,d^{2b}},

with :math:`a` and :math:`b` set by UMAP's ``min_dist`` of 0.1, the closest that two points are packed. Training draws linked pairs :math:`(i, j)` with probability proportional to :math:`w_{ij}`, and five random points :math:`k` for each, and minimises

.. math::

   \mathcal{L} = -\log q(d_{ij}) - \frac{1}{5} \sum_{k} \log\bigl(1 - q(d_{ik})\bigr),

pulling linked points together and pushing random points apart.
