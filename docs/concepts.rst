Concepts
========

alphaUniverse runs every galaxy through AION :cite:p:`parker2025aion`, a :term:`foundation model` trained on more than 200 million observations of stars, galaxies and quasars from Legacy Survey, HSC, SDSS, DESI and Gaia. alphaUniverse uses the released model, ``polymathic-ai/aion-base``, without retraining it.

AION works in two stages. Each observation is first cut into small pieces, and each piece is replaced by an integer, a :term:`token`. A :term:`transformer` then reads all of a galaxy's tokens together. Everything alphaUniverse shows comes from two things it produces: an :term:`embedding` of each token, and a :term:`prediction` of what every token should be.

.. raw:: html
   :file: figures/aion-pipeline.svg

From observations to tokens
---------------------------

Each kind of observation has its own :term:`codec`, a small network trained to compress it into tokens and back. A codec has a fixed vocabulary of :term:`codes <code>`; pieces of data that look alike get the same code.

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

Each position a token can occupy is a :term:`slot`: one cell of one survey's image, one 25.6 Å window of one survey's spectrum, one catalogue value, or the redshift. A galaxy has 1,722 slots in all: 576 per image survey, 272 per spectrum survey, 25 catalogue values and the redshift. A spectrum's flux normalisation token sets its overall scale; the encoder reads it, but it is never predicted or searched, so it has no slot.

The slots fall into seven :term:`modes <mode>`, one per kind of observation from one survey: the Legacy Survey and HSC images, the DESI and SDSS spectra, the Legacy Survey and HSC catalogue values, and the redshift.

Embeddings
----------

AION's encoder reads all of a galaxy's observed tokens at once and returns a vector of 768 numbers for each. Because each token is read alongside all the others, its *encoded* embedding depends on the whole galaxy, not only on its own code. alphaUniverse stores these, and the *codebook* embeddings AION starts from, which depend only on the code. A galaxy's :term:`mean embedding`, the average of its encoded embeddings, is the single vector that the map uses.

Two embeddings are compared by the :term:`cosine similarity` of their directions: 1 when they point the same way, 0 when they are unrelated.

What AION predicts
------------------

AION was trained by hiding some of a galaxy's tokens and predicting them from the rest. alphaUniverse asks it to predict every slot of every galaxy from the galaxy's own tokens: at each slot :math:`s` it gives a probability distribution :math:`p_{g,s}(t)` over the codes :math:`t`. A sharply peaked distribution means AION is confident which code belongs there; a flat one means it is not. It does this even for observations the galaxy lacks, so a galaxy without a DESI spectrum still has a predicted one. :doc:`methods/similarity-search` compares galaxies through these distributions.
