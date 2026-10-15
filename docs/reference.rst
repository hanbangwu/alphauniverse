Reference
=========

How it fits together
--------------------

The build pipeline and the app share nothing but files. Eleven Modal jobs write artifacts to a volume mounted at ``/cache``, under ``$ALPHAUNIVERSE_CACHE/<author>/<name>/<revision>/``, so changing ``DATASET_REVISION`` switches trees rather than overwriting one. The FastAPI app in ``app/main.py`` reads them in one Modal container; the SvelteKit frontend, deployed on Vercel, calls it. ``main`` deploys to Modal through ``.github/workflows/ci-cd.yml``.

============================ ====================================== ==================================
Job                          Reads                                  Writes
============================ ====================================== ==================================
generate_embeddings          the dataset                            encoded, codebook, tokens
generate_predictions         tokens                                 probabilities, probability_basis
generate_token_grams         probabilities, probability_basis       token_gram_<mode>,
                                                                    token_gram_<mode>_anywhere
generate_token_embeddings    token_gram_<mode>,                     token_embedding_<mode>,
                             token_gram_<mode>_anywhere             token_embedding_<mode>_anywhere
generate_templates           probabilities, probability_basis       template_<mode>
generate_observed_grams      the dataset, tokens, template_<mode>   data_gram_<mode>,
                                                                    data_gram_<mode>_anywhere,
                                                                    template_gram_<mode>,
                                                                    template_gram_<mode>_anywhere,
                                                                    observed_reference
generate_template_embeddings probabilities, probability_basis,      template_embedding_<mode>,
                             template_gram_<mode>,                  template_embedding_<mode>_anywhere
                             template_gram_<mode>_anywhere,
                             observed_reference
generate_projections         the dataset, encoded                   mean_points, full_points,
                                                                    parametric_umap
generate_pairs               the dataset, encoded                   pairs_<modality>
generate_alignment           pairs_<modality>                       alignment
generate_aion_gemma_space    pairs_<modality>, alignment            aion_gemma_space
============================ ====================================== ==================================

Every artifact with one row per galaxy holds them in dataset row order, and the app relies on it without checking, so a build directory must come from one complete run. Each job commits the volume before it returns. The model and codecs load from the latest commit of ``polymathic-ai/aion-base``. The serving app reads images, spectra and the catalogue from the dataset cached on the volume, with ``HF_HUB_OFFLINE=1``, which loads the most recently cached revision whatever ``DATASET_REVISION`` says, so the volume must hold only the revision the artifacts were built from.

Artifacts
---------

``encoded``, ``codebook`` and ``tokens`` share one layout: a ``galaxy`` column, one cell per token survey and the redshift (a list of 768-dimensional ``float16`` embeddings, or of ``uint32`` token ids for ``tokens``, null where unmatched), and a ``bool`` per flag survey. Within an image cell the image tokens come first, then the survey's catalogue values; a spectrum cell leads with its normalisation token. ``encoded`` and ``tokens`` are uncompressed Arrow IPC files that readers memory-map; ``codebook`` is Parquet.

.. autofunction:: app.config.store_schema
   :no-index:

``probabilities`` is an uncompressed Arrow IPC file of AION's predictions at every slot. Image tokens keep their 64 most probable codes with their probabilities, the mass beyond the first 16 and the mass beyond all 64; search reads each galaxy's 16 most probable and spreads the rest evenly over the other codes, but the query galaxy's full 64; catalogue values and the redshift keep every probability; spectrum tokens keep 256 coefficients on a per-survey PCA basis, quantised to 8 bits as ``offset + code × step``.

.. autodata:: app.config.PREDICTIONS
   :no-index:
   :no-value:

``probability_basis`` is an ``np.savez`` of each spectrum survey's mean (``float64``, 1024) and top 256 eigenvectors (``float64``, 256 × 1024) of the spectrum token probabilities; a token's distribution is ``mean + coefficients @ directions``.

``token_gram_<mode>`` and ``token_gram_<mode>_anywhere`` are ``np.save`` files of a galaxies × galaxies ``float32`` :term:`Gram matrix`: one for each mode at the same place, and one for each image and spectrum mode at any position. Entry :math:`(q, g)` is the mode score :math:`T_m(g)` of :doc:`methods/similarity-search` with every slot of the mode selected on :math:`q` and every galaxy read as search reads the query galaxy, so a same-place matrix is symmetric up to rounding. The products run in TF32 with float32 accumulation. The app does not read them.

``token_embedding_<mode>`` and ``token_embedding_<mode>_anywhere`` are ``np.savez`` files of :term:`token embeddings <token embedding>`, one per Gram matrix :math:`K`. With :math:`K_c` the matrix less its row and column means, and :math:`s` and :math:`V` its top 1,024 singular values and right singular vectors: ``queries`` is :math:`K_c V s^{-1/2}` and ``galleries`` is :math:`V s^{1/2}`, galaxies × 1,024 ``float32`` each; ``offsets`` holds :math:`K`'s column means and ``self_scores`` its diagonal :math:`K(g, g)`, in ``float32``; ``singular_values`` holds all of :math:`K_c`'s in ``float64``, as square roots of :math:`K_c^\top K_c`'s eigenvalues, so values below about :math:`10^{-8}` of the largest are rounding error. ``queries[q] @ galleries.T + offsets`` ranks the galaxies as row :math:`q` of :math:`K` does, exactly at full rank.

``template_<mode>`` is an ``np.save`` file of every galaxy's :term:`template` in the mode, ``float32``, one row per galaxy, observed in the mode or not. An image template is AION's codec decoding of the expected latent at every cell, :math:`\sum_t p_{g,s}(t)\, z(t)` with the mass beyond the stored codes spread evenly over the rest of the vocabulary, 4 Legacy Survey or 5 HSC bands × 96 × 96 px in the survey's flux units: the job decodes every band in Legacy Survey units and multiplies HSC's by :math:`10^{1.8}`, the inverse of the codec's scaling from HSC's zero point of 27. A spectrum template is the decoding of the expected latent at every spectral window without the normalisation token, 8,704 values on the codec's grid from 3,500 Å in 0.8 Å steps: a shape, since the normalisation token is not predicted. A catalogue value's or the redshift's template is the median of its predicted distribution over the codec's bins, each at the value AION decodes it to, its upper edge. The open top bin takes its lower edge instead: AION decodes it to the largest value in the codec's sample of its training data (1.9 × 10⁹ nmgy for ``FLUX_Z``) or to no value. Bins past it decode to no value and are skipped. Decoding runs in IEEE float32.

``data_gram_<mode>`` and ``data_gram_<mode>_anywhere`` are ``np.save`` files of :term:`against-data scores <against-data score>`, ``float32``, one row per galaxy and one column per galaxy observed in the mode with data on the template's pixels, in the order of ``observed_reference``'s ``<mode>_galleries``. With :math:`\mu_q` the query's :term:`template`, :math:`\mu_0` the median of all galaxies' templates, and :math:`x_g` and :math:`w_g` the gallery galaxy's data and inverse variances on the template's pixels (the central 96 × 96 px of an image; for a spectrum, the codec's grid, through the transpose of the linear interpolation to each observed wavelength; for catalogue values and the redshift, the errors of ``template_quality``):

.. math::

   S(q, g) = \langle \mu_q - \mu_0,\; w_g x_g \rangle - \tfrac12 \langle \mu_q^2 - \mu_0^2,\; w_g \rangle = S_\text{sat}(g) - \tfrac12 \chi^2(x_g; \mu_q)

with :math:`\chi^2(x_g; \mu) = \langle (x_g - \mu)^2,\; w_g \rangle` and :math:`S_\text{sat}(g) = \tfrac12 \chi^2(x_g; \mu_0)` the gallery galaxy's :term:`saturation`. A spectrum template takes its best amplitude, :math:`S(q, g) = \tfrac12 \left[ \langle \mu_q, w_g x_g \rangle^2 / \langle \mu_q^2, w_g \rangle - \langle \mu_0, w_g x_g \rangle^2 / \langle \mu_0^2, w_g \rangle \right]`. At any position, each of the query's :term:`slots <slot>` (a 4 × 4 px cell in every band, or one of the 32-pixel windows the survey covers) takes its best score over the gallery galaxy's slots with data, each against :math:`\mu_0` at that slot, and the slot scores add, so :math:`S` can exceed :math:`S_\text{sat}`; a spectrum's data are first divided by the best amplitude of the galaxy's own template. Same-place scores are computed in float64, any-position ones in IEEE float32.

``template_gram_<mode>`` and ``template_gram_<mode>_anywhere`` are ``np.save`` files of galaxies × galaxies ``float32`` :term:`template Grams <template Gram>`. Each template is scaled to the mode's :term:`typical depth`, :math:`a_g = \mu_g \sqrt{\bar w}`, with :math:`\bar w` the median of :math:`w_g` over the gallery galaxies; for a spectrum, :math:`w_g` is in units of the galaxy's own template's amplitude and :math:`a_g` is scaled to unit length. Entry :math:`(q, g)` is :math:`-\tfrac12 \lVert a_q - a_g \rVert^2`, or at any position the sum over the query's slots of :math:`-\tfrac12 \lVert a_{q,s} - a_{g,s'} \rVert^2` at the gallery galaxy's nearest slot :math:`s'`. As for the scores, same-place entries are computed in float64 and any-position ones in IEEE float32.

``observed_reference`` is an ``np.savez`` holding, for each mode, ``<mode>_reference`` (:math:`\mu_0`), ``<mode>_galleries`` (the galaxies observed in the mode with data on the template's pixels), ``<mode>_saturation`` (each one's :term:`saturation`), ``<mode>_depth`` (:math:`\sqrt{\bar w}`) and, for spectra, ``<mode>_amplitudes`` (each galaxy's own template's best amplitude on its data).

``template_embedding_<mode>`` and ``template_embedding_<mode>_anywhere`` are :term:`template embeddings <template embedding>`, ``np.savez`` files with the parts of a token embedding, decomposed from the template Grams, and ``widths``, each galaxy's :term:`predictive width` in the mode: for an image or spectrum the root mean square over slots of the variance of AION's predicted codec latent, summed over its dimensions; for catalogue values and the redshift the root mean square over values of half the width of the predicted central 68%, times :math:`\sqrt{\bar w}`.

``mean_points`` (one row per galaxy) and ``full_points`` (one row per embedding, written survey by survey) are Parquet:

.. autodata:: app.config.POINTS
   :no-index:
   :no-value:

``pairs_image`` and ``pairs_spectrum`` are Parquet, in galaxy order. ``pairs_image`` holds, for every galaxy, the mean of AION's Legacy Survey image embeddings and EmbeddingGemma's embedding of the colour image; ``pairs_spectrum`` holds, for every galaxy with a DESI spectrum, the mean of AION's DESI spectrum embeddings and EmbeddingGemma's embedding of the spectrum plot:

.. autodata:: app.config.PAIRS
   :no-index:
   :no-value:

``parametric_umap`` and ``alignment`` are ``torch.save`` files of the projector's and the text maps' weights; ``alignment`` holds, for each modality, the network and the linear map the text search benchmark compares. ``aion_gemma_space`` is an ``np.savez`` of the rows of ``pairs_image`` then ``pairs_spectrum`` carried into EmbeddingGemma's space: ``vectors`` (``float32``, rows × 768, unit length), ``galaxy`` (``int32``) and ``modality`` (``image`` or ``spectrum``).

HTTP API
--------

Every endpoint is ``GET``; ``/projections/{projection}`` and ``/downloads/{role}`` also answer ``HEAD``. A request whose ``If-None-Match`` matches the response's ETag gets ``304 Not Modified``. Artifact files carry Starlette's ETag, from size and modification time; every other response's ETag is an MD5 of its body. Every response carries ``Cache-Control: no-cache``.

The first Same place Population map (``/saliency``) on a galaxy builds its agreement matrix, the log overlap of every galaxy at every slot, and the server keeps the last 8. An Any position map builds no matrix: it computes the rows of its sample of galaxies, described under ``/saliency`` below, and the server keeps them for the last 8 pairs of galaxy and Any position modes. These computations, the matrix builds and the sums of a new Any position search run one at a time.

.. openapi:: ../frontend/openapi.json

Tests and CI
------------

Tests run on a synthetic artifact tree that ``scripts/fixture.py`` builds in the production schemas, with AION and its codecs at random weights; no test downloads weights. ``/galaxy/{galaxy}/image``, ``/galaxy/{galaxy}/spectrum``, EmbeddingGemma, ``generate_pairs``, ``generate_alignment`` and ``generate_aion_gemma_space`` are untested, ``generate_token_grams`` is tested on the CPU in float32 rather than TF32, and the frontend has no tests.

Continuous integration lints and tests every pull request, checks that ``frontend/openapi.json`` is current and builds these docs with warnings as errors. On ``main``, once lint, tests and the ``openapi.json`` check pass, it deploys to Modal.
