The map
=======

The map places every galaxy in two dimensions, next to the galaxies similarity search ranks as most alike. Only closeness carries meaning. The axes have no units, and the gap between distant clusters says little, since two dimensions cannot keep every distance :cite:p:`mcinnes2018umap`.

For each galaxy, alphaUniverse runs a :term:`whole-mode search` over all seven modes and keeps the 14 best matches. **Same place** compares every mode at the same place; **Any position** compares images and spectra at any position. UMAP takes the galaxy and these matches as its 15 neighbours, at distance :math:`1 + s_1 - s_k`, where :math:`s_k` is the :math:`k`-th neighbour's score, and lays the galaxies out from that graph alone. Search scores are not distances, so UMAP uses only their order and spread around each galaxy.
