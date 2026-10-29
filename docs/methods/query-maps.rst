Query maps
==========

Before or after a search, the query galaxy's image and spectrum can be shaded to show what a selection picks out. **Population** also colours the catalogue values.

This galaxy
-----------

**This galaxy** shades each slot :math:`c` of an image or spectrum mode :math:`m` by how alike AION predicts it is to the selection in that mode: :math:`\max_{s \in S_m} \cos(p_{q,s}, p_{q,c})`. A selected slot scores 1, and other slots score high where AION expects the same kind of content. Catalogue values are not shaded, since each has its own codes.

Population
----------

**Population** asks: across the field, where does agreeing with :math:`q` go with matching the selection? For every other galaxy :math:`g` it takes

- :math:`T(g) = \sum_{s \in S} \log O_s(q, g)`, its score on the selection :math:`S` over every mode, unstandardised;
- :math:`A_c(g) = \log O_c(q, g)`, its agreement with :math:`q` at slot :math:`c`;
- :math:`C(g)`, its mean agreement with :math:`q` outside the selection.

A galaxy that resembles :math:`q` everywhere would make :math:`A_c` and :math:`T` correlate at every slot. The :term:`partial correlation` removes the part :math:`C` explains:

.. math::

   \rho_c = \frac{r(A_c, T) - r(A_c, C)\, r(T, C)}
   {\sqrt{\bigl(1 - r(A_c, C)^2\bigr)\bigl(1 - r(T, C)^2\bigr)}},

with :math:`r` the Pearson correlation over galaxies. A high :math:`\rho_c` marks a slot whose agreement predicts a good match beyond overall resemblance: select one emission line and the galaxy's other lines and its redshift light up. Slots next to the selection light up too, because neighbouring predictions are alike.

.. raw:: html
   :file: ../figures/query-maps.svg

With **Any position**, :math:`A_c`, :math:`T` and :math:`C` use the best overlap anywhere in the other galaxy's image or spectrum, over a fixed random sample of 512 galaxies.
