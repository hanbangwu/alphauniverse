Similarity search
=================

You pick a galaxy :math:`q` and a selection: image cells, spectrum tokens, catalogue values, or any mix. The selected slots of each :term:`mode` :math:`m` form a set :math:`S_m`. alphaUniverse then ranks every other galaxy :math:`g` by how well AION's predictions for it agree with its predictions for :math:`q` at those slots (``app.pql.search``).

Why predictions, not embeddings
-------------------------------

An encoded embedding records where its token sits and which survey it comes from, as well as what the token shows, so comparing embeddings partly compares which observations two galaxies have. AION's predictions avoid this: it predicts every slot of every galaxy, observed or not, as a distribution over the same codes. Two galaxies are then compared on what AION expects at each slot, and a galaxy that lacks a selected observation is still ranked on what AION expects it to look like; the match list says so.

.. raw:: html
   :file: ../figures/similarity-search.svg

The score
---------

At one slot the agreement is the *overlap*, the probability that a code drawn from each prediction is the same:

.. math::

   O_s(q, g) = \sum_t p_{q,s}(t)\, p_{g,s}(t).

This is the expected likelihood kernel :cite:p:`jebara2004probability`. It is symmetric, has no fitted parameters, and is high only when both predictions put their weight on the same codes: two confident predictions that agree score highest, two that disagree score lowest, and a broad, uncertain prediction scores in between.

A mode's score adds the log overlaps of its slots, treating each slot as independent evidence:

.. math::

   T_m(g) = \sum_{s \in S_m} \log O_s(q, g).

Neighbouring image cells and spectral windows are not independent, so a selection of many adjacent slots counts some evidence more than once.

A plain sum over modes would let the mode with the most selected slots set the ranking: 30 image cells would outweigh one redshift. Each mode's score is therefore standardised over all galaxies, with mean :math:`\langle T_m \rangle` and standard deviation :math:`\sigma(T_m)`, and the :math:`M` standardised scores are averaged:

.. math::

   \text{score}(g) = \frac{1}{M} \sum_{m=1}^{M}
   \frac{T_m(g) - \langle T_m \rangle}{\sigma(T_m)}.

The similarity
--------------

The **similarity** shown beside each match puts the score on a scale from 0 to 1. For exact distributions an overlap can be no larger than the query's own most probable code, :math:`\hat p_{q,s} = \max_t p_{q,s}(t)`, so

.. math::

   \ell_m(g) = \frac{1}{|S_m|} \sum_{s \in S_m}
   \bigl(\log O_s(q, g) - \log \hat p_{q,s}\bigr) \le 0,

and a mode's similarity :math:`e^{\ell_m}` is the geometric mean fraction of the best agreement any galaxy could reach. The overall similarity is :math:`\exp \sum_m w_m \ell_m`, with :math:`w_m \propto 1/\sigma(\ell_m)` and :math:`\sum_m w_m = 1`; it orders galaxies exactly as the score does. The stored predictions are compressed (:doc:`/reference`), so a similarity can slightly exceed 1; the site shows at most 1.

Each match also shows a **map**: at every slot :math:`c`, :math:`O_c(q, g) / \hat p_{q,c}`, the same fraction slot by slot, so you can see where in the image or spectrum the two galaxies agree.

Same place or any position
--------------------------

By default a selected image cell is compared with the same cell of the other galaxy, and a spectrum token with the same observed wavelength. With **Any position**, each selected slot is compared with every slot of the other galaxy's image or spectrum, and the best is kept:

.. math::

   \log O_s^{\text{any}}(q, g) = \max_{s'} \log \sum_t p_{q,s}(t)\, p_{g,s'}(t).

This is the maximum-similarity matching of ColBERT :cite:p:`khattab2020colbert`. It finds a feature wherever it sits, at the cost of the selection's layout: each selected slot finds its best match on its own, so two selected emission lines can match at two different redshifts. The match's map then shows, at each slot, the best fraction any selected slot reaches there, and outlines where each selected slot found its best match.

Whole modes
-----------

With every slot of a mode selected, :math:`T_m` is close to a row of the mode's :term:`Gram matrix`, which reads the other galaxies' predictions as search reads the query's (:doc:`/reference`). A :term:`whole-mode search` ranks from the mode's :term:`token embedding` instead: :math:`q`'s query vector times every galaxy's gallery vector, both cut to their first 256 components, plus the offsets, gives that row up to a constant and the components left out. Each chosen mode's scores, at any position for an image or spectrum mode set to it, are standardised over all galaxies and averaged as above; the constant cancels. A match shows each mode's standardised score divided by :math:`M`, its contribution, so the contributions sum to the score, and the same-place map above at every slot.
