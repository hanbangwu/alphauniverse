Query maps
==========

Before you search, the query galaxy's panels can shade its own slots, to show what a selection picks out. Both maps cover the image and spectrum; **Population** also colours the table.

This galaxy
-----------

**This galaxy** shows, at each slot :math:`c` of an image or spectrum mode :math:`m`, how alike AION predicts it is to the selection in that mode: the best cosine between predicted distributions, :math:`\max_{s \in S_m} \cos(p_{q,s}, p_{q,c})` (``app.pql.cosines``). A selected slot scores 1 against itself, and other slots score high where AION expects the same kind of content. Catalogue values are not shaded, since each has its own vocabulary of codes.

Population
----------

**Population** asks a different question: across the whole field, where does agreeing with :math:`q` go with matching the selection? It uses every other galaxy :math:`g` and three quantities:

- :math:`T(g) = \sum_{s \in S} \log O_s(q, g)`, the galaxy's score on the selection :math:`S`, the union of the selected slots of every mode, summed without the standardisation search uses;
- :math:`A_c(g) = \log O_c(q, g)`, its agreement with :math:`q` at slot :math:`c`;
- :math:`C(g)`, its mean agreement with :math:`q` outside the selection.

A galaxy that resembles :math:`q` everywhere agrees at every slot and matches the selection too, so :math:`A_c` and :math:`T` would correlate at every slot. The :term:`partial correlation` removes that shared part, the part :math:`C` explains:

.. math::

   \rho_c = \frac{r(A_c, T) - r(A_c, C)\, r(T, C)}
   {\sqrt{\bigl(1 - r(A_c, C)^2\bigr)\bigl(1 - r(T, C)^2\bigr)}},

where :math:`r` is the Pearson correlation over galaxies (``app.pql.saliency``). A high :math:`\rho_c` marks a slot whose agreement predicts a good match beyond overall resemblance: an emission line selected in the spectrum lights up the galaxy's other lines and its redshift when they move together across the field. Slots next to the selection light up too, because neighbouring slots' predictions are alike.

.. raw:: html
   :file: ../figures/query-maps.svg

With Any position, :math:`A_c`, :math:`T` and :math:`C` use the best overlap anywhere in the other galaxy's image or spectrum, and the correlation runs over 512 other galaxies drawn at random, the same for every selection on the query galaxy, since the best overlap at every slot of every galaxy would take too long to compute.
