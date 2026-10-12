The map
=======

The map places each embedding in two dimensions so that embeddings that are neighbours in 768 dimensions stay neighbours on the screen. **Mean** shows one point per galaxy, the mean of its token embeddings; **full** shows one point per token embedding, every modality in the same space.

The projection is a *parametric UMAP*: a small neural network :math:`f_\theta: \mathbb{R}^{768} \to \mathbb{R}^2` trained on 70% of 5,000,000 embeddings drawn at random. UMAP first weights each pair of near neighbours in 768 dimensions by cosine distance, :math:`w_{ij}`. On the map, two points at distance :math:`d` count as neighbours with probability

.. math::

   q(d) = \frac{1}{1 + a\,d^{2b}},

with :math:`a` and :math:`b` set by UMAP's ``min_dist`` of 0.1. Training draws pairs :math:`(i, j)` with probability proportional to :math:`w_{ij}` and five random points :math:`k`, and minimises

.. math::

   \mathcal{L} = -\log q(d_{ij}) - \frac{1}{5} \sum_{k} \log\bigl(1 - q(d_{ik})\bigr),

pulling neighbours together and pushing random points apart.
