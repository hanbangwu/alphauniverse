Glossary
========

.. glossary::
   :sorted:

   against-data score
      The log-likelihood of one galaxy's data under another galaxy's :term:`template`, with the data's own errors, less its log-likelihood under the median template. See :doc:`reference`.

   predictive width
      How uncertain AION's prediction of a galaxy is in one :term:`mode`: the spread of the predicted codec latent, or of a predicted catalogue value. See :doc:`reference`.

   saturation
      A galaxy's highest possible :term:`against-data score` in a :term:`mode`: the score of a template equal to its data.

   template embedding
      A galaxy's query and gallery vectors for one :term:`mode`, from the singular value decomposition of the mode's :term:`template Gram`, as for a :term:`token embedding`.

   template Gram
      A :term:`mode`'s :term:`templates <template>` compared with each other at the :term:`typical depth`: minus half the squared distance between every pair. See :doc:`reference`.

   typical depth
      The median inverse variance of each pixel or value over the galaxies with data in a :term:`mode`.

   breadth
      How widely a text query spreads over :math:`N` galaxies: :math:`\exp(H)/N`, with :math:`H` the entropy of its probabilities over them. See :doc:`benchmarks`.

   classical MDS
      Classical multidimensional scaling: each galaxy's coordinates :math:`E_g` along the eigenvectors of a double-centred symmetric matrix, scaled by the square roots of their eigenvalues, so that :math:`E_q \cdot E_g` reproduces the matrix when none of its eigenvalues is negative.

   code
      One entry in a :term:`codec`'s fixed vocabulary. A :term:`token` is the code of one piece of an observation.

   codec
      A small network that turns one kind of observation (an image, a spectrum, a catalogue value) into :term:`tokens <token>` and back. See :doc:`concepts`.

   cosine similarity
      The cosine of the angle between two vectors: 1 when they point the same way, 0 when they are perpendicular, −1 when they point opposite ways. It ignores their lengths.

   embedding
      The vector of 768 numbers AION's encoder returns for a :term:`token`. The encoded embedding depends on all of the galaxy's tokens; the codebook embedding depends only on the token's :term:`code`.

   EmbeddingGemma
      Google's model that places images and text in one 768-dimensional space, so that a picture and a description of it lie close together. Text search uses it. See :doc:`methods/text-search`.

   foundation model
      A large network trained once on broad data and then used unchanged for many tasks.

   Gram matrix
      A :term:`mode`'s search scores between every pair of galaxies: entry :math:`(q, g)` scores galaxy :math:`g` against query :math:`q` with every :term:`slot` of the mode selected. See :doc:`reference`.

   held out
      Left out of a network's training and used only to test it.

   joint embedding
      One vector per galaxy from its :term:`token embeddings <token embedding>` or :term:`template embeddings <template embedding>`: each mode's first 256 gallery components, scaled to unit root-mean-square length about their mean, concatenated over the modes: 2,816 values for the 11 blocks.

   mean embedding
      The average of a galaxy's encoded :term:`embeddings <embedding>`, one vector per galaxy.

   mode
      One kind of observation from one survey: the Legacy Survey or HSC image, the DESI or SDSS spectrum, the Legacy Survey or HSC catalogue values, or the redshift. Similarity search scores each mode separately.

   parametric UMAP
      A small network trained to place embeddings on a two-dimensional map so that embeddings that are neighbours in 768 dimensions stay neighbours :cite:p:`sainburg2021parametric`. ``generate_projections`` trains one for ``mean_points`` and ``full_points`` (:doc:`reference`).

   partial correlation
      The correlation between two quantities after removing, from each, the part a third quantity explains linearly. See :doc:`methods/query-maps`.

   prediction
      The probability AION assigns to each :term:`code` at a :term:`slot`, given the galaxy's tokens. See :doc:`concepts`.

   PROVABGS
      A catalogue of stellar masses, star formation rates and redshifts fitted to DESI spectra and photometry. alphaUniverse uses it as ground truth for the probe benchmark, its masses and star formation rates for the text search benchmark, and in ``properties`` as the measured mass and star formation rate, to which the predicted ones are fitted.

   slot
      A position a :term:`token` can occupy: one image cell, one spectral window, one catalogue value or the redshift. A galaxy has 1,722.

   sSFR
      Specific star formation rate: star formation rate divided by stellar mass, in yr\ :sup:`-1`.

   template
      A galaxy's :term:`prediction` in one :term:`mode` turned back into an observation by the mode's :term:`codec`: an image, a spectrum's shape, or a catalogue value or redshift. See :doc:`reference`.

   token
      The integer a :term:`codec` assigns to one piece of an observation, such as a 4 × 4 px image cell or a 25.6 Å spectral window.

   token embedding
      A galaxy's query and gallery vectors for one :term:`mode`, from the singular value decomposition of the mode's :term:`Gram matrix`: :math:`q`'s query vector times :math:`g`'s gallery vector, plus an offset for :math:`g`, ranks :math:`g` for query :math:`q` as search does, up to the components left out. Unrelated to the :term:`embedding` of a token.

   transformer
      A network that reads a sequence of tokens and lets the output for each depend on every other token in the sequence.

   whole-mode search
      Search over every slot of each chosen :term:`mode`, ranked from the modes' :term:`token embeddings <token embedding>`. See :doc:`methods/similarity-search`.
