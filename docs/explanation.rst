How alphaUniverse works
=======================

AION is a transformer trained on large surveys of galaxies, stars, and quasars across imaging, spectroscopy, and catalogues. All data is first turned into a sequence of discrete *tokens*, and AION learns which tokens go together. Everything alphaUniverse shows comes from two things AION produces for each galaxy: an *embedding* of each token, and a *prediction* of what every token should be.

From observations to tokens
---------------------------

Each modality has its own *codec*, a small network that compresses an observation into tokens, each an integer from a fixed vocabulary of *codes*.

==================== ========================================== ==============
Observation          Tokens                                     Codes
==================== ========================================== ==============
Image                24 × 24 grid over the 96 × 96 px cutout,   4375 per token
                     one per 4 × 4 px cell (576)
Spectrum             one per 25.6 Å from 3500 Å (272), plus one 1024 per token
                     for the flux normalisation
Catalogue value      one per value: 12 Legacy Survey, 13 HSC    1024
Redshift             one                                        1025, up to 6
==================== ========================================== ==============

A galaxy has 1,722 slots in all: 576 per image survey, 272 per spectrum survey, 25 catalogue values and the redshift.

Embeddings
----------

AION's encoder reads all of a galaxy's tokens at once and returns a 768-dimensional vector for each. alphaUniverse stores these *encoded* embeddings, and the *codebook* embeddings AION starts from, which depend only on the code and not on the galaxy.

The map
-------

The map places each embedding in two dimensions so that embeddings that are neighbours in 768 dimensions stay neighbours on the screen. **Mean** shows one point per galaxy, the mean of its token embeddings; **full** shows one point per token embedding, every modality in the same space.

The projection is a *parametric UMAP*: a small neural network :math:`f_\theta: \mathbb{R}^{768} \to \mathbb{R}^2` trained on 70% of 5,000,000 embeddings drawn at random. UMAP first weights each pair of near neighbours in 768 dimensions by cosine distance, :math:`w_{ij}`. On the map, two points at distance :math:`d` count as neighbours with probability

.. math::

   q(d) = \frac{1}{1 + a\,d^{2b}},

with :math:`a` and :math:`b` set by UMAP's ``min_dist`` of 0.1. Training draws pairs :math:`(i, j)` with probability proportional to :math:`w_{ij}` and five random points :math:`k`, and minimises

.. math::

   \mathcal{L} = -\log q(d_{ij}) - \frac{1}{5} \sum_{k} \log\bigl(1 - q(d_{ik})\bigr),

pulling neighbours together and pushing random points apart.

What AION predicts
------------------

AION was trained by hiding some of a galaxy's tokens and predicting them from the rest. alphaUniverse asks it to predict every slot of every galaxy from the galaxy's own tokens: at each slot :math:`s` it gives a probability distribution :math:`p_{g,s}(t)` over the codes :math:`t`. It does this even for observations the galaxy lacks, so a galaxy without a DESI spectrum still has a predicted one.

Similarity search
-----------------

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
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default a selected image cell is compared with the same cell of the other galaxy, and a spectrum token with the same observed wavelength. With **Any position**, each selected slot is compared with every slot of the other galaxy's image or spectrum, and the best is kept:

.. math::

   \log O_s^{\text{any}}(q, g) = \max_{s'} \log \sum_t p_{q,s}(t)\, p_{g,s'}(t).

This finds a feature wherever it sits, at the cost of position. The match's map then shows, at each slot, the best fraction any selected slot reaches there, and outlines where each selected slot found its best match.

Query maps
----------

Before you search, the query galaxy's panels can shade its own slots.

**This galaxy** shows, at each slot :math:`c` of an image or spectrum mode :math:`m`, how alike AION predicts it is to the selection in that mode: the best cosine between predicted distributions, :math:`\max_{s \in S_m} \cos(p_{q,s}, p_{q,c})` (``app.pql.cosines``). Catalogue values are not shaded.

**Population** asks a different question: across the whole field, where does agreeing with :math:`q` go with matching the selection? For each slot :math:`c` it takes, over every other galaxy, the partial correlation between the galaxy's selection score :math:`S(g) = \sum_{s \in S} \log O_s(q, g)` and its agreement there, :math:`A_c(g) = \log O_c(q, g)`, controlling for its mean agreement outside the selection, :math:`C(g)`:

.. math::

   \rho_c = \frac{r(A_c, S) - r(A_c, C)\, r(S, C)}
   {\sqrt{\bigl(1 - r(A_c, C)^2\bigr)\bigl(1 - r(S, C)^2\bigr)}},

where :math:`r` is the Pearson correlation over galaxies (``app.pql.saliency``). A high :math:`\rho_c` marks a slot whose agreement predicts a good match beyond the galaxies simply resembling :math:`q` overall. With Any position, the correlation runs over the 256 best matches and 256 random other galaxies.

Text search
-----------

Text search uses a second model, `EmbeddingGemma <https://huggingface.co/google/embeddinggemma-2>`_, which places images and text in one 768-dimensional space where a picture and a description of it lie close together.

For each galaxy, alphaUniverse pairs two vectors: AION's mean embedding :math:`a_g`, and EmbeddingGemma's embedding :math:`e_g` of its Legacy Survey colour image. A small network

.. math::

   f(a) = \frac{W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2}
   {\lVert W_2\, \mathrm{GELU}(W_1 a + b_1) + b_2 \rVert},
   \qquad W_1 \in \mathbb{R}^{2048 \times 768},

is trained on 70% of the galaxies to minimise :math:`1 - \cos(f(a_g), e_g)`, carrying AION's view of each galaxy into EmbeddingGemma's space. A query is embedded by EmbeddingGemma as :math:`t`, and galaxies are ranked by :math:`\cos(f(a_g), t)` (``app.text_search.text_search``). :doc:`benchmarks` measures how well these rankings recover PROVABGS properties.

What the panels show
--------------------

The image is the Legacy Survey colour cutout, the central 96 × 96 px. The spectrum is DESI's, smoothed for display with a Gaussian of σ = 2 pixels; search uses AION's predictions, never the smoothed flux. The table lists every catalogue value of the galaxy; the 26 AION reads can be selected for search. Morphologies are Galaxy Zoo 10 labels, and most galaxies have none.
