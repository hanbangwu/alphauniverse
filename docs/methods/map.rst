The map
=======

The map places galaxies in two dimensions so that neighbours stay neighbours on the screen. **Same place** and **Any position**, one point per galaxy, take each galaxy's neighbours under search (below). **Mean** shows one point per galaxy, its :term:`mean embedding`; **Full** shows one point per token embedding, every mode in the same space; both project AION's embeddings, neighbours in 768 dimensions.

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

Maps placed by search
---------------------

``generate_maps`` (``app.maps``) places each galaxy by its neighbours under :term:`whole-mode search` over all seven modes: ``same_place_points`` at the same place, ``any_position_points`` with the images and spectra at any position. For every galaxy it scores every other galaxy with ``/search/modes``' own scoring (``app.pql.whole_mode_totals``): the stored token embeddings at 256 components, each mode standardised over all galaxies and averaged. It keeps the 14 highest. UMAP takes these as each galaxy's 15 neighbours, itself included, at distance :math:`1 + s_1 - s_k`, where :math:`s_k` is the :math:`k`-th neighbour's score, and lays the galaxies out from that graph alone; the joint embeddings only give it one row per galaxy. Search scores need not satisfy the triangle inequality, so these are not distances in any embedding; UMAP uses only their order and spread around each galaxy. ``neighbour_quality`` measured how many of each galaxy's search neighbours such a map keeps (:doc:`/benchmarks`).
