Concepts
========

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

What AION predicts
------------------

AION was trained by hiding some of a galaxy's tokens and predicting them from the rest. alphaUniverse asks it to predict every slot of every galaxy from the galaxy's own tokens: at each slot :math:`s` it gives a probability distribution :math:`p_{g,s}(t)` over the codes :math:`t`. It does this even for observations the galaxy lacks, so a galaxy without a DESI spectrum still has a predicted one.
