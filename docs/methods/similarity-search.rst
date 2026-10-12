Similarity search
=================

You pick a galaxy :math:`q` and a selection: image cells, spectrum tokens, catalogue values, or any mix. Each kind is a *mode* :math:`m` with selected slots :math:`S_m`. alphaUniverse then ranks every other galaxy :math:`g` by how well AION's predictions for it agree with its predictions for :math:`q` at those slots (``app.pql.search``).

At one slot the agreement is the *overlap*, the probability that a code drawn from each prediction is the same:

.. math::

   O_s(q, g) = \sum_t p_{q,s}(t)\, p_{g,s}(t).

A mode's score adds the log overlaps of its slots, as if the slots were independent evidence:

.. math::

   S_m(g) = \sum_{s \in S_m} \log O_s(q, g).

With several modes, each score is standardised over all galaxies and the standardised scores are averaged, so no mode wins by having more slots:

.. math::

   \text{score}(g) = \frac{1}{M} \sum_{m=1}^{M}
   \frac{S_m(g) - \langle S_m \rangle}{\sigma(S_m)}.

The **similarity** shown beside each match rescales this to between 0 and 1. An overlap can be no larger than the query's own most probable code, :math:`\hat p_{q,s} = \max_t p_{q,s}(t)`, so

.. math::

   \ell_m(g) = \frac{1}{|S_m|} \sum_{s \in S_m}
   \bigl(\log O_s(q, g) - \log \hat p_{q,s}\bigr) \le 0,

and a mode's similarity :math:`e^{\ell_m}` is the geometric mean fraction of the best agreement any galaxy could reach. The overall similarity is :math:`\exp \sum_m w_m \ell_m`, with :math:`w_m \propto 1/\sigma(\ell_m)` and :math:`\sum_m w_m = 1`; it orders galaxies as the score does.

Because the comparison uses predictions, a galaxy that lacks a selected observation is still ranked, on what AION expects it to look like; the match list says so.

Each match also shows a **map**: at every slot :math:`c`, :math:`O_c(q, g) / \hat p_{q,c}`, the same fraction slot by slot, so you can see where in the image or spectrum the two galaxies agree.

Same place or any position
--------------------------

By default a selected image cell is compared with the same cell of the other galaxy, and a spectrum token with the same observed wavelength. With **Any position**, each selected slot is compared with every slot of the other galaxy's image or spectrum, and the best is kept:

.. math::

   \log O_s^{\text{any}}(q, g) = \max_{s'} \log \sum_t p_{q,s}(t)\, p_{g,s'}(t).

This finds a feature wherever it sits, at the cost of position. The match's map then shows, at each slot, the best fraction any selected slot reaches there, and outlines where each selected slot found its best match.
