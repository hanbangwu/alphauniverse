Concepts
========

alphaUniverse runs every galaxy through AION :cite:p:`parker2025aion`, a :term:`foundation model`. AION was trained once on more than 200 million observations of stars, galaxies and quasars from Legacy Survey, HSC, SDSS, DESI and Gaia, and alphaUniverse uses it without retraining.

AION works in two steps. It first compresses each observation into a short list of integers, its :term:`tokens <token>`, much as a lossy image format stores an image in fewer numbers. A :term:`transformer` then reads all of a galaxy's tokens together. alphaUniverse uses two of its outputs: an :term:`embedding` of each token, and a :term:`prediction` of what every token should be.

.. raw:: html
   :file: figures/aion-pipeline.svg

Tokens
------

Each kind of observation has its own :term:`codec`, a small network that compresses it into tokens and decompresses them. A codec has a fixed vocabulary of :term:`codes <code>`, like a palette: pieces of data that look alike get the same code.

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

Each place a token can occupy is a :term:`slot`: one cell of one survey's image, one 25.6 Å window of one survey's spectrum, one catalogue value, or the redshift. A galaxy has 1,722 slots: 576 per image survey, 272 per spectrum survey, 25 catalogue values and the redshift. The normalisation token sets a spectrum's overall flux; AION reads it but never predicts it, so it has no slot.

The slots fall into seven :term:`modes <mode>`, one per kind of observation from one survey: the Legacy Survey and HSC images, the DESI and SDSS spectra, the Legacy Survey and HSC catalogue values, and the redshift.

Embeddings
----------

For each token AION returns 768 numbers, its embedding, which describe the token in the context of the whole galaxy: the same code gets a different embedding in a different galaxy. Two embeddings are compared by their :term:`cosine similarity`: 1 when they point the same way, 0 when they are unrelated. A galaxy's :term:`mean embedding` averages its tokens'.

Predictions
-----------

AION was trained by hiding some of a galaxy's tokens and predicting them from the rest. alphaUniverse asks it to predict every slot of every galaxy: at slot :math:`s` of galaxy :math:`g`, a probability :math:`p_{g,s}(t)` for each code :math:`t`. A peaked distribution means AION is confident; a flat one means it is not. AION also predicts observations the galaxy lacks, so a galaxy without a DESI spectrum still has a predicted one. Decompressing a prediction through the codec gives a :term:`template`, what AION expects the observation to look like. :doc:`methods/similarity-search` compares galaxies through these predictions.
