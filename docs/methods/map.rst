The map
=======

The map places galaxies in two dimensions so that neighbours stay neighbours on the screen. **Any position** and **Same place** show one point per galaxy, placed by its neighbours under search (below).

Only closeness on the map carries meaning. The axes have no units, and the distance between two clusters far apart says little about how different they are; a projection to two dimensions cannot keep every distance, and this one is built to keep neighbours :cite:p:`mcinnes2018umap`. :doc:`/benchmarks` measures how many neighbours it keeps.

How it is built
---------------

``generate_maps`` (``app.maps``) places each galaxy by its neighbours under :term:`whole-mode search` over all seven modes: ``same_place_points`` at the same place, ``any_position_points`` with the images and spectra at any position. For every galaxy it scores every other galaxy with ``/search/modes``' own scoring (``app.pql.whole_mode_totals``): the stored token embeddings at 256 components, each mode standardised over all galaxies and averaged. It keeps the 14 highest. UMAP takes these as each galaxy's 15 neighbours, itself included, at distance :math:`1 + s_1 - s_k`, where :math:`s_k` is the :math:`k`-th neighbour's score, and lays the galaxies out from that graph alone; the joint embeddings only give it one row per galaxy. Search scores need not satisfy the triangle inequality, so these are not distances in any embedding; UMAP uses only their order and spread around each galaxy. ``neighbour_quality`` measured how many of each galaxy's search neighbours such a map keeps (:doc:`/benchmarks`).
