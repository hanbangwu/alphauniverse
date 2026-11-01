Similarity search
=================

You pick a galaxy :math:`q` and a selection: image cells, spectrum windows, catalogue values, or any mix. The selected slots of each :term:`mode` :math:`m` form a set :math:`S_m`. alphaUniverse ranks every other galaxy :math:`g` by how well AION's predictions for it agree with its predictions for :math:`q` at those slots.

Why predictions, not embeddings
-------------------------------

An embedding records which survey and position its token comes from as well as what it shows, so comparing embeddings partly compares which observations two galaxies have. AION predicts every slot of every galaxy, observed or not, over the same codes, so comparing predictions compares only what AION expects at each slot. A galaxy that lacks a selected observation is ranked on AION's prediction of it, and the match list says so.

.. raw:: html
   :file: ../figures/similarity-search.svg

The score
---------

At one slot the agreement is the *overlap*, the probability that a code drawn from each prediction is the same :cite:p:`jebara2004probability`:

.. math::

   O_s(q, g) = \sum_t p_{q,s}(t)\, p_{g,s}(t).

Two confident predictions that agree score highest, two that disagree score lowest, and an uncertain prediction scores in between. A mode's score adds the log overlaps of its slots:

.. math::

   T_m(g) = \sum_{s \in S_m} \log O_s(q, g).

Neighbouring slots are not independent, so many adjacent slots count some evidence more than once.

So that 30 image cells do not outweigh one redshift, each mode's score is standardised over all galaxies and the :math:`M` modes are averaged:

.. math::

   \text{score}(g) = \frac{1}{M} \sum_{m=1}^{M}
   \frac{T_m(g) - \langle T_m \rangle}{\sigma(T_m)}.

The similarity
--------------

The **similarity** beside each match puts the score on a scale from 0 to 1. An overlap is at most the query's most probable code, :math:`\hat p_{q,s} = \max_t p_{q,s}(t)`, so

.. math::

   \ell_m(g) = \frac{1}{|S_m|} \sum_{s \in S_m}
   \bigl(\log O_s(q, g) - \log \hat p_{q,s}\bigr) \le 0,

and :math:`e^{\ell_m}` is the geometric mean fraction of the best agreement any galaxy could reach. The overall similarity is :math:`\exp \sum_m w_m \ell_m`, with :math:`w_m \propto 1/\sigma(\ell_m)` and :math:`\sum_m w_m = 1`, which orders galaxies as the score does. The stored predictions are compressed, so a similarity can slightly exceed 1; the site shows at most 1.

Each match's **map** shows :math:`O_c(q, g) / \hat p_{q,c}` at every slot :math:`c`: where in the image or spectrum the two galaxies agree.

Same place or any position
--------------------------

By default a selected image cell is compared with the same cell of the other galaxy, and a spectrum window with the same observed wavelength. With **Any position**, each selected slot is compared with every slot of the other galaxy's image or spectrum and keeps the best :cite:p:`khattab2020colbert`:

.. math::

   \log O_s^{\text{any}}(q, g) = \max_{s'} \log \sum_t p_{q,s}(t)\, p_{g,s'}(t).

This finds a feature wherever it sits, but each selected slot matches on its own, so two selected emission lines can match at two different redshifts. The map then outlines where each selected slot found its best match.

Whole modes
-----------

A :term:`whole-mode search` selects every slot of each chosen mode. It reads each mode's score from a precomputed summary of every pair of galaxies, which matches :math:`T_m` up to a constant that standardisation cancels. Each match shows each mode's standardised score divided by :math:`M`, its contribution, so the contributions sum to the score.
