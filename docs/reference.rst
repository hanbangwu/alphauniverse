Reference
=========

How it fits together
--------------------

The build pipeline and the app share nothing but files. Seventeen Modal jobs write artifacts to a volume mounted at ``/cache``, under ``$ALPHAUNIVERSE_CACHE/<author>/<name>/<revision>/``, so changing ``DATASET_REVISION`` switches trees rather than overwriting one. The FastAPI app in ``app/main.py`` reads them in one Modal container; the SvelteKit frontend, deployed on Vercel, calls it. ``main`` deploys to Modal through ``.github/workflows/ci-cd.yml``.

============================ ====================================== ==================================
Job                          Reads                                  Writes
============================ ====================================== ==================================
generate_embeddings          the dataset                            encoded, codebook, tokens
generate_predictions         tokens                                 probabilities, probability_basis
generate_token_grams         probabilities, probability_basis       token_gram_<mode>,
                                                                    token_gram_<mode>_anywhere
generate_token_embeddings    token_gram_<mode>,                     token_embedding_<mode>,
                             token_gram_<mode>_anywhere             token_embedding_<mode>_anywhere,
                                                                    token_search_<mode>,
                                                                    token_search_<mode>_anywhere
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
generate_joint_embedding     token_embedding_<mode>,                joint_embedding
                             token_embedding_<mode>_anywhere
generate_properties          the dataset, probabilities,            properties
                             joint_embedding
generate_maps                the dataset, joint_embedding,          same_place_points,
                             token_search_<mode>,                   any_position_points
                             token_search_<mode>_anywhere
generate_projections         the dataset, encoded                   mean_points, full_points,
                                                                    parametric_umap
generate_pairs               the dataset, encoded                   pairs_<modality>
generate_alignment           pairs_<modality>                       alignment
generate_aion_gemma_space    pairs_<modality>, alignment            aion_gemma_space
generate_text_views          the dataset, tokens, probabilities,    text_views, text_views_held_out
                             probability_basis, template_<mode>
train_text_flow              the dataset, tokens, probabilities,    text_flow, text_bases, text_views,
                             probability_basis, template_<mode>,    text_views_held_out
                             joint_embedding, text_views,
                             text_views_held_out
generate_text_clouds         text_flow, text_bases,                 text_clouds, text_query_mean
                             joint_embedding
============================ ====================================== ==================================

Every artifact with one row per galaxy holds them in dataset row order, and the app relies on it without checking, so a build directory must come from one complete run. Each job commits the volume before it returns. The model and codecs load from the latest commit of ``polymathic-ai/aion-base``. Each spectrum reaches the codec followed by one masked sample at wavelength 99,999 with zero flux and inverse variance, which gives the tokens of the codec on AION's ``main`` branch: it pads every spectrum with such samples, to 7,808 for DESI and 4,800 for SDSS, and only the first affects the tokens. AION 0.0.2 does not pad. The serving app reads images, spectra and the catalogue from the dataset cached on the volume, with ``HF_HUB_OFFLINE=1``, which loads the most recently cached revision whatever ``DATASET_REVISION`` says, so the volume must hold only the revision the artifacts were built from.

Artifacts
---------

``encoded``, ``codebook`` and ``tokens`` share one layout: a ``galaxy`` column, one cell per token survey and the redshift (a list of 768-dimensional ``float16`` embeddings, or of ``uint32`` token ids for ``tokens``, null where unmatched), and a ``bool`` per flag survey. Within an image cell the image tokens come first, then the survey's catalogue values; a spectrum cell leads with its normalisation token. ``encoded`` and ``tokens`` are uncompressed Arrow IPC files that readers memory-map; ``codebook`` is Parquet.

.. autofunction:: app.config.store_schema
   :no-index:

``probabilities`` is an uncompressed Arrow IPC file of AION's predictions at every slot. Image tokens keep their 64 most probable codes with their probabilities, the mass beyond the first 16 and the mass beyond all 64; search reads each galaxy's 16 most probable and spreads the rest evenly over the other codes, but the query galaxy's full 64; catalogue values and the redshift keep every probability; spectrum tokens keep 256 coefficients on a per-survey PCA basis, quantised to 8 bits as ``offset + code × step``.

.. autodata:: app.config.PREDICTIONS
   :no-index:
   :no-value:

``probability_basis`` is an ``np.savez`` of each spectrum survey's mean (``float64``, 1024) and top 256 eigenvectors (``float64``, 256 × 1024) of the spectrum token probabilities; a token's distribution is ``mean + coefficients @ directions``. ``generate_predictions`` fits the coefficients against it in ``float64``, and every later prediction must too; search decodes them in ``float32``.

``token_gram_<mode>`` and ``token_gram_<mode>_anywhere`` are ``np.save`` files of a galaxies × galaxies ``float32`` :term:`Gram matrix`: one for each mode at the same place, and one for each image and spectrum mode at any position. Entry :math:`(q, g)` is the mode score :math:`T_m(g)` of :doc:`methods/similarity-search` with every slot of the mode selected on :math:`q` and every galaxy read as search reads the query galaxy, so a same-place matrix is symmetric up to rounding. The products run in TF32 with float32 accumulation. The app does not read them.

``token_embedding_<mode>`` and ``token_embedding_<mode>_anywhere`` are ``np.savez`` files of :term:`token embeddings <token embedding>`, one per Gram matrix :math:`K`. With :math:`K_c` the matrix less its row and column means, and :math:`s` and :math:`V` its top 1,024 singular values and right singular vectors: ``queries`` is :math:`K_c V s^{-1/2}` and ``galleries`` is :math:`V s^{1/2}`, galaxies × 1,024 ``float32`` each; ``offsets`` holds :math:`K`'s column means and ``self_scores`` its diagonal :math:`K(g, g)`, in ``float32``; ``singular_values`` holds all of :math:`K_c`'s in ``float64``, as square roots of :math:`K_c^\top K_c`'s eigenvalues, so values below about :math:`10^{-8}` of the largest are rounding error. ``queries[q] @ galleries.T + offsets`` ranks the galaxies as row :math:`q` of :math:`K` does, exactly at full rank. A new query's row :math:`k` of scores against every galaxy has the query vector :math:`(k - \bar k - o + \bar o)\, V s^{-1/2}`, with :math:`o` the offsets; for a galaxy's own row this is its stored query vector.

``token_search_<mode>`` and ``token_search_<mode>_anywhere``, from the same job, are ``np.savez`` files of the same ``queries`` and ``galleries`` cut to their first 256 components, and ``offsets``. The app loads them at startup for :term:`whole-mode search`, 392 MB for 17,369 galaxies (11 × 17,369 × (2 × 256 + 1) × 4 bytes), and it does not start without them.

``template_<mode>`` is an ``np.save`` file of every galaxy's :term:`template` in the mode, ``float32``, one row per galaxy, observed in the mode or not. An image template is AION's codec decoding of the expected latent at every cell, :math:`\sum_t p_{g,s}(t)\, z(t)` with the mass beyond the stored codes spread evenly over the rest of the vocabulary, 4 Legacy Survey or 5 HSC bands × 96 × 96 px in the survey's flux units: the job decodes every band in Legacy Survey units and multiplies HSC's by :math:`10^{1.8}`, the inverse of the codec's scaling from HSC's zero point of 27. A spectrum template is the decoding of the expected latent at every spectral window without the normalisation token, 8,704 values on the codec's grid from 3,500 Å in 0.8 Å steps: a shape, since the normalisation token is not predicted. A catalogue value's or the redshift's template is the median of its predicted distribution over the codec's bins, each at the value AION decodes it to, its upper edge. The open top bin takes its lower edge instead: AION decodes it to the largest value in the codec's sample of its training data (1.9 × 10⁹ nmgy for ``FLUX_Z``) or to no value. Bins past it decode to no value and are skipped. Decoding runs in IEEE float32.

``text_views`` is a directory of Parquet parts, ``part-<n>.parquet``, of 4,000 :term:`views <view>` each, drawn from the galaxies outside text search's held-out set: the 30% held out of the text maps and the prototype's 143, which leaves 12,058. A view has one of 26 types: the observation, the :term:`template` or a sample in each :term:`mode`, or one of 5 combinations of modes in a single input. Each part draws its (galaxy, view type) pairs from its own seed, weighted 1 per template, observation and combination, 4 per image or spectrum sample and 8 per catalogue sample. A sample draws every slot's token from AION's stored distribution, the mass beyond an image cell's stored codes spread evenly over the rest, and decodes it with the mode's codec. An image is turned by one of its 8 rotations and reflections; a spectrum is smoothed by a Gaussian of σ = 1, 2 or 3 px and plotted at 1,680 × 384 px over the survey's observed range, its flux axis from the 0.5th to the 99.9th percentile with 10% padding; catalogue values are written one ``COLUMN: value`` line each, shuffled, at 3, 4 or 5 significant figures. A combination draws each member as observation, template or sample, the observation only where one exists. ``round`` numbers each ten parts. A rerun keeps the parts already written, so a stopped job resumes. ``view_type`` is the type's position in ``app.config.VIEW_TYPES``:

.. autodata:: app.config.TEXT_VIEWS
   :no-index:
   :no-value:

``text_views_held_out`` holds the views ``train_text_flow`` checks the flow against: one of every type available to each of 500 :term:`held out` galaxies, the prototype's 143 among them, drawn from a random stream of their own:

.. autodata:: app.config.TEXT_VIEWS_HELD_OUT
   :no-index:
   :no-value:

``text_bases`` is an ``np.savez`` of the coordinates ``train_text_flow`` fits on the first round of ``text_views``. Per view type, ``mean`` is the mean of its embeddings (26 × 768), and a 32-component PCA of its mean-subtracted unit embeddings gives ``components`` (26 × 32 × 768), ``centre`` (26 × 768) and ``scale`` (26 × 32), each component's standard deviation; a type with fewer than 128 views in the round takes its PCA from its mode's three types together. ``offset`` and ``projection`` whiten ``joint_embedding`` to 64 components over the training galaxies, and ``kappa`` is :term:`κ`. ``text_flow`` is a ``torch.save`` of the flow's weights: a network for the velocity that carries Gaussian noise to a view's 32 coefficients, given its type and the whitened joint embedding. In training, each of the joint embedding's 11 blocks is dropped with probability 0.3, and all of them with probability 0.1. It trains on the stored views 400 at a time, 60 steps of 512 views each, drawn so that every view type a galaxy has is as likely as any other, and keeps an exponential moving average of its weights. Every 4,000 views it measures the :term:`identification gain` of the held-out views at κ of 5, 10, 15, 20, 30 and 40, and it stops once 40,000 views have added less than 0.01 nats, keeping the weights and κ of the last check that added more; it first runs ``generate_text_views`` for the first round and the held-out views, and again for another round whenever the stored views run out, up to 1,000,000 views. Its report, which adds the gain with each mode's blocks dropped or alone and that of a flow from views to the whitened joint embedding, trained on the same views and scored by its log-density ratios scaled by each of 0.01, 0.02, 0.05, 0.1, 0.25, 0.5 and 1, goes to Weights & Biases and the job's log.

``text_clouds`` is an ``np.save`` file of every galaxy's :term:`cloud`, ``float16``, view types × galaxies × 16 × 32: each point's 32 coefficients on its type's ``components``, times its ``scale``, after 32 Euler steps of ``text_flow`` from noise seeded by the galaxy, with every block of its joint embedding present. The app reads it in batches into ``float32`` on the first text search and computes each point's length :math:`\lVert c_k + \sum_j x_j U_{k,j} \rVert` from ``text_bases``. ``text_query_mean`` is an ``np.save`` file of the mean EmbeddingGemma embedding of the 10,080 generic queries (``float32``, 768), which every query has subtracted before scoring.

``data_gram_<mode>`` and ``data_gram_<mode>_anywhere`` are ``np.save`` files of :term:`against-data scores <against-data score>`, ``float32``, one row per galaxy and one column per galaxy observed in the mode with data on the template's pixels, in the order of ``observed_reference``'s ``<mode>_galleries``. With :math:`\mu_q` the query's :term:`template`, :math:`\mu_0` the median of all galaxies' templates, and :math:`x_g` and :math:`w_g` the gallery galaxy's data and inverse variances on the template's pixels (the central 96 × 96 px of an image; for a spectrum, the codec's grid, through the transpose of the linear interpolation to each observed wavelength; for catalogue values and the redshift, the errors of ``template_quality``):

.. math::

   S(q, g) = \langle \mu_q - \mu_0,\; w_g x_g \rangle - \tfrac12 \langle \mu_q^2 - \mu_0^2,\; w_g \rangle = S_\text{sat}(g) - \tfrac12 \chi^2(x_g; \mu_q)

with :math:`\chi^2(x_g; \mu) = \langle (x_g - \mu)^2,\; w_g \rangle` and :math:`S_\text{sat}(g) = \tfrac12 \chi^2(x_g; \mu_0)` the gallery galaxy's :term:`saturation`. A spectrum template takes its best amplitude, :math:`S(q, g) = \tfrac12 \left[ \langle \mu_q, w_g x_g \rangle^2 / \langle \mu_q^2, w_g \rangle - \langle \mu_0, w_g x_g \rangle^2 / \langle \mu_0^2, w_g \rangle \right]`. At any position, each of the query's :term:`slots <slot>` (a 4 × 4 px cell in every band, or one of the 32-pixel windows the survey covers) takes its best score over the gallery galaxy's slots with data, each against :math:`\mu_0` at that slot, and the slot scores add, so :math:`S` can exceed :math:`S_\text{sat}`; a spectrum's data are first divided by the best amplitude of the galaxy's own template. Same-place scores are computed in float64, any-position ones in IEEE float32.

``template_gram_<mode>`` and ``template_gram_<mode>_anywhere`` are ``np.save`` files of galaxies × galaxies ``float32`` :term:`template Grams <template Gram>`. Each template is scaled to the mode's :term:`typical depth`, :math:`a_g = \mu_g \sqrt{\bar w}`, with :math:`\bar w` the median of :math:`w_g` over the gallery galaxies; for a spectrum, :math:`w_g` is in units of the galaxy's own template's amplitude and :math:`a_g` is scaled to unit length. Entry :math:`(q, g)` is :math:`-\tfrac12 \lVert a_q - a_g \rVert^2`, or at any position the sum over the query's slots of :math:`-\tfrac12 \lVert a_{q,s} - a_{g,s'} \rVert^2` at the gallery galaxy's nearest slot :math:`s'`. As for the scores, same-place entries are computed in float64 and any-position ones in IEEE float32.

``observed_reference`` is an ``np.savez`` holding, for each mode, ``<mode>_reference`` (:math:`\mu_0`), ``<mode>_galleries`` (the galaxies observed in the mode with data on the template's pixels), ``<mode>_saturation`` (each one's :term:`saturation`), ``<mode>_depth`` (:math:`\sqrt{\bar w}`) and, for spectra, ``<mode>_amplitudes`` (each galaxy's own template's best amplitude on its data).

``template_embedding_<mode>`` and ``template_embedding_<mode>_anywhere`` are :term:`template embeddings <template embedding>`, ``np.savez`` files with the parts of a token embedding, decomposed from the template Grams, and ``widths``, each galaxy's :term:`predictive width` in the mode: for an image or spectrum the root mean square over slots of the variance of AION's predicted codec latent, summed over its dimensions; for catalogue values and the redshift the root mean square over values of half the width of the predicted central 68%, times :math:`\sqrt{\bar w}`.

``joint_embedding`` is an ``np.save`` file of every galaxy's :term:`joint embedding` from its token embeddings, galaxies × 2,816 ``float32``: the 11 blocks' 256 columns each, in ``GRAM_NAMES`` order, so any block can be taken alone.

``properties`` is a Parquet file of every galaxy's redshift, stellar mass (``mass``, :math:`\log_{10} M_\odot`), star formation rate (``sfr``, :math:`\log_{10} M_\odot\,\text{yr}^{-1}`), specific star formation rate (``ssfr``, :math:`\log_{10} \text{yr}^{-1}`), :term:`half-light radius` (``radius``, arcseconds), r magnitude (``r_magnitude``), :term:`axis ratio` (``axis_ratio``), Hα :term:`equivalent width` (``halpha_ew``, Å) and :term:`Dn4000` (``dn4000``), each measured where it has a value and predicted for every galaxy, with the prediction's width. The measured values are:

- the spectroscopic redshift where usable, DESI's before SDSS's;
- :term:`PROVABGS`'s mass and star formation rate;
- from the Legacy Survey's model of the galaxy: ``SHAPE_R``; :math:`22.5 - 2.5 \log_{10}` of the r flux in nanomaggies, null where the flux is not positive; and the axis ratio, null where both ellipticity components are 0, as in models round by construction;
- from the spectrum of the survey whose redshift is used, moved to the rest frame: Hα's equivalent width over 6,554.6 to 6,574.6 Å, against the mean of the median fluxes over 6,490 to 6,530 Å and 6,600 to 6,640 Å, positive for emission; and the Dn4000, the mean of :math:`f_\lambda \lambda^2` over 4,000 to 4,100 Å divided by its mean over 3,850 to 3,950 Å. Each band must lie inside the spectrum's unmasked pixels with at least 5 of them, and the Dn4000's 3,850 to 3,950 Å mean must exceed three times its error. So Hα is measured up to redshift 9,824 / 6,640 − 1 = 0.48 in DESI and 9,221.5 / 6,640 − 1 = 0.39 in SDSS.

The redshift's prediction is the median of AION's predicted distribution, and its width half the range from the distribution's 16th to its 84th percentile. Both spread each bin's probability evenly over the redshifts its token covers, from the bin below up to its own value: a distribution in one bin, 0.0059 wide, predicts the bin's centre with a width of 0.0020, where the redshift :term:`template` takes the bin's top. A galaxy with a measured redshift read it, so its prediction is close to it; ``redshift_quality`` measures predictions without it. The others come from ridge regression on the joint embedding, its penalty chosen by leave-one-out, fitted to the galaxies with a measured value. Those galaxies' own predictions come from 5-fold cross-validation, so none is fitted to its own value, and the RMS error of those predictions is the width, the same for every galaxy. PROVABGS fits DESI spectra, so its widths are measured only on galaxies DESI observed, and the spectral ones only on galaxies with a spectrum. ``g_r`` is the observed Legacy Survey colour, :math:`-2.5 \log_{10}` of the g to r flux ratio, null where either flux is not positive. ``/downloads/properties`` serves it.

.. autodata:: app.config.PROPERTIES
   :no-index:
   :no-value:

``mean_points`` (one row per galaxy) and ``full_points`` (one row per embedding, written survey by survey) are Parquet, and so are ``same_place_points`` and ``any_position_points``, one row per galaxy placed by :term:`whole-mode search` over every mode (:doc:`methods/map`):

.. autodata:: app.config.POINTS
   :no-index:
   :no-value:

``pairs_image`` and ``pairs_spectrum`` are Parquet, in galaxy order. ``pairs_image`` holds, for every galaxy, the mean of AION's Legacy Survey image embeddings and EmbeddingGemma's embedding of the colour image; ``pairs_spectrum`` holds, for every galaxy with a DESI spectrum, the mean of AION's DESI spectrum embeddings and EmbeddingGemma's embedding of the spectrum plot:

.. autodata:: app.config.PAIRS
   :no-index:
   :no-value:

``parametric_umap`` and ``alignment`` are ``torch.save`` files of the projector's and the text maps' weights; ``alignment`` holds, for each modality, the network and a linear map fitted to the same pairs as a baseline. ``aion_gemma_space`` is an ``np.savez`` of the rows of ``pairs_image`` then ``pairs_spectrum`` carried into EmbeddingGemma's space: ``vectors`` (``float32``, rows × 768, unit length), ``galaxy`` (``int32``) and ``modality`` (``image`` or ``spectrum``).

HTTP API
--------

Every endpoint is ``GET``; ``/projections/{projection}`` and ``/downloads/{role}`` also answer ``HEAD``. A request whose ``If-None-Match`` matches the response's ETag gets ``304 Not Modified``. Artifact files carry Starlette's ETag, from size and modification time; every other response's ETag is an MD5 of its body. Every response carries ``Cache-Control: no-cache``.

The first Same place Population map (``/saliency``) on a galaxy builds its agreement matrix, the log overlap of every galaxy at every slot, and the server keeps the last 8. An Any position map builds no matrix: it computes the rows of its sample of galaxies, described under ``/saliency`` below, and the server keeps them for the last 8 pairs of galaxy and Any position modes. These computations, the matrix builds and the sums of a new Any position search run one at a time.

.. openapi:: ../frontend/openapi.json

Tests and CI
------------

Tests run on a synthetic artifact tree that ``scripts/fixture.py`` builds in the production schemas, with AION and its codecs at random weights; no test downloads weights. ``/galaxy/{galaxy}/image/{survey}``, ``/galaxy/{galaxy}/spectrum/{survey}``, EmbeddingGemma, ``generate_pairs``, ``generate_alignment``, ``generate_aion_gemma_space`` and the observed views of ``generate_text_views`` are untested, ``generate_token_grams`` is tested on the CPU in float32 rather than TF32, ``generate_maps`` is tested for its neighbours but not its UMAP layout, which needs more galaxies than UMAP's 15 neighbours, and the frontend has no tests.

Continuous integration lints and tests every pull request, checks that ``frontend/openapi.json`` is current and builds these docs with warnings as errors. On ``main``, once lint, tests and the ``openapi.json`` check pass, it deploys to Modal's default environment. Running the workflow by hand on any branch deploys that branch to the ``dev`` environment, which has its own app, volume and secrets. Vercel builds every branch: production builds from ``main`` and calls the default app, and preview builds from every other branch call the ``dev`` app, whose URL ``PUBLIC_API_URL`` holds in each Vercel environment. The ``dev`` app runs whichever branch last ran the workflow, so previews share it.
