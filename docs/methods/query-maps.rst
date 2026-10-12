Query maps
==========

Before you search, the query galaxy's panels can shade its own slots.

**This galaxy** shows, at each slot :math:`c` of an image or spectrum mode :math:`m`, how alike AION predicts it is to the selection in that mode: the best cosine between predicted distributions, :math:`\max_{s \in S_m} \cos(p_{q,s}, p_{q,c})` (``app.pql.cosines``). Catalogue values are not shaded.

**Population** asks a different question: across the whole field, where does agreeing with :math:`q` go with matching the selection? For each slot :math:`c` it takes, over every other galaxy, the partial correlation between the galaxy's selection score :math:`S(g) = \sum_{s \in S} \log O_s(q, g)` and its agreement there, :math:`A_c(g) = \log O_c(q, g)`, controlling for its mean agreement outside the selection, :math:`C(g)`:

.. math::

   \rho_c = \frac{r(A_c, S) - r(A_c, C)\, r(S, C)}
   {\sqrt{\bigl(1 - r(A_c, C)^2\bigr)\bigl(1 - r(S, C)^2\bigr)}},

where :math:`r` is the Pearson correlation over galaxies (``app.pql.saliency``). A high :math:`\rho_c` marks a slot whose agreement predicts a good match beyond the galaxies simply resembling :math:`q` overall. With Any position, the correlation runs over the 256 best matches and 256 random other galaxies.
