Glossary
========

.. glossary::
   :sorted:

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

   held out
      Left out of a network's training and used only to test it.

   mean embedding
      The average of a galaxy's encoded :term:`embeddings <embedding>`, one vector per galaxy.

   mode
      One kind of observation from one survey: the Legacy Survey or HSC image, the DESI or SDSS spectrum, the Legacy Survey or HSC catalogue values, or the redshift. Similarity search scores each mode separately.

   parametric UMAP
      A small network trained to place embeddings on a two-dimensional map so that embeddings that are neighbours in 768 dimensions stay neighbours. See :doc:`methods/map`.

   partial correlation
      The correlation between two quantities after removing, from each, the part a third quantity explains linearly. See :doc:`methods/query-maps`.

   prediction
      The probability AION assigns to each :term:`code` at a :term:`slot`, given the galaxy's tokens. See :doc:`concepts`.

   PROVABGS
      A catalogue of stellar masses, star formation rates and redshifts fitted to DESI spectra and photometry. alphaUniverse uses it as ground truth for the text search benchmark.

   slot
      A position a :term:`token` can occupy: one image cell, one spectral window, one catalogue value or the redshift. A galaxy has 1,722.

   sSFR
      Specific star formation rate: star formation rate divided by stellar mass, in yr\ :sup:`-1`.

   template
      A galaxy's :term:`prediction` in one :term:`mode` turned back into an observation by the mode's :term:`codec`: an image, a spectrum's shape, or a catalogue value or redshift. See :doc:`reference`.

   token
      The integer a :term:`codec` assigns to one piece of an observation, such as a 4 × 4 px image cell or a 25.6 Å spectral window.

   transformer
      A network that reads a sequence of tokens and lets the output for each depend on every other token in the sequence.
