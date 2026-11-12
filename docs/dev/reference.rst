Reference
=========

How it fits together
--------------------

The build pipeline and the app share nothing but files. Sixteen Modal jobs write artifacts to the root of a volume mounted at ``/cache``, so changing ``DATASET_REVISION`` overwrites them. ``app.config.BUILD_DIR`` is ``/cache`` on Modal and ``.cache`` elsewhere, a copy of all or part of the volume. The FastAPI app in ``app/main.py`` reads them in one Modal container; the SvelteKit web app, deployed on Vercel, calls it through a rewrite of ``/api``. ``main`` deploys to Modal through ``.github/workflows/ci-cd.yml``.

=================================== =========================================== =========================================
Job                                 Reads                                       Writes
=================================== =========================================== =========================================
generate_embeddings                 the dataset                                 encoded, tokens
generate_predictions                tokens                                      probabilities, probability_basis
generate_token_grams                probabilities, probability_basis            token_gram_<mode>,
                                                                                token_gram_<mode>_anywhere
generate_token_embeddings           token_gram_<mode>,                          token_embedding_<mode>,
                                    token_gram_<mode>_anywhere                  token_embedding_<mode>_anywhere,
                                                                                token_search_<mode>,
                                                                                token_search_<mode>_anywhere
generate_aion_predictions           probabilities, probability_basis            aion_prediction_<mode>
generate_observed_grams             the dataset, tokens, aion_prediction_<mode> data_gram_<mode>,
                                                                                data_gram_<mode>_anywhere,
                                                                                aion_prediction_gram_<mode>,
                                                                                aion_prediction_gram_<mode>_anywhere,
                                                                                observed_reference
generate_aion_prediction_embeddings probabilities, probability_basis,           aion_prediction_embedding_<mode>,
                                    aion_prediction_gram_<mode>,                aion_prediction_embedding_<mode>_anywhere
                                    aion_prediction_gram_<mode>_anywhere,
                                    observed_reference
generate_joint_embedding            token_embedding_<mode>,                     joint_embedding
                                    token_embedding_<mode>_anywhere
generate_properties                 the dataset, probabilities,                 properties
                                    joint_embedding
generate_maps                       the dataset, joint_embedding,               same_place_points,
                                    token_search_<mode>,                        any_position_points
                                    token_search_<mode>_anywhere
generate_patterns                   encoded                                     patterns
generate_projections                the dataset, encoded                        mean_points, full_points,
                                                                                parametric_umap
generate_pairs                      the dataset, encoded                        pairs_<modality>
generate_text_views                 the dataset, tokens, probabilities,         text_views, text_views_held_out
                                    probability_basis, aion_prediction_<mode>
train_text_regression               the dataset, tokens, probabilities,         text_regression, text_bases,
                                    probability_basis, aion_prediction_<mode>,  text_views, text_views_held_out
                                    joint_embedding, text_views,
                                    text_views_held_out
generate_text_vectors               text_regression, text_bases,                text_vectors, text_query_mean
                                    joint_embedding
=================================== =========================================== =========================================

Every artifact with one row per galaxy holds them in dataset row order, and the app relies on it without checking, so a build directory must come from one complete run. Each job commits the volume before it returns. The model and codecs load from the latest commit of ``polymathic-ai/aion-base``. Each spectrum reaches the codec followed by one masked sample at wavelength 99,999 with zero flux and inverse variance, which gives the tokens of the codec on AION's ``main`` branch: it pads every spectrum with such samples, to 7,808 for DESI and 4,800 for SDSS, and only the first affects the tokens. AION 0.0.2 does not pad. The serving app reads images, spectra and the catalogue from the dataset cached on the volume, with ``HF_HUB_OFFLINE=1``, which loads the most recently cached revision whatever ``DATASET_REVISION`` says, so the volume must hold only the revision the artifacts were built from.

``scripts/dev_subset.py`` builds the dev subset: a Modal function draws ``--galaxies`` dataset rows at random with ``SEED``, clears ``.dev_subset`` on the volume and writes there the dataset's rows and every artifact cut to them, renumbered from 0 in dataset order; the script then replaces the local ``.cache`` with it. Grams keep the chosen rows and columns, data Grams the columns of the chosen galleries, and ``observed_reference`` the chosen galleries. Production's values stay: offsets, singular values, maps, ``probability_basis``, ``text_bases``, ``text_query_mean`` and the model weights come from every galaxy, and ``PROTOTYPE_HELD_OUT`` names production rows.

Features in code
----------------

=========================== ========================================
Feature                     Code
=========================== ========================================
Similarity search           ``app.pql.search``
Whole-mode search           ``app.pql.mode_search``
This galaxy map             ``app.pql.cosines``
Population map              ``app.pql.saliency``
Text search                 ``app.text_search.text_search``
The map                     ``app.maps``, run by ``generate_maps``
=========================== ========================================

A :term:`whole-mode search` ranks from the mode's :term:`token embedding`: :math:`q`'s query vector times every galaxy's gallery vector, both from ``token_search_<mode>`` at 256 components, plus the offsets, gives the mode's :term:`Gram matrix` row up to a constant and the components left out. Each chosen mode's scores, at any position for an image or spectrum mode set to it, are standardised over all galaxies, one mode's too, and averaged.

``generate_maps`` scores every galaxy against every other with ``app.pql.whole_mode_totals``, the scoring ``/search/modes`` uses, and writes ``same_place_points`` and ``any_position_points``. The joint embeddings only give UMAP one row per galaxy; the layout comes from the neighbour graph (:doc:`/methods/map`).

Artifacts
---------

``encoded`` and ``tokens`` share one layout: a ``galaxy`` column, one cell per token survey and the redshift (a list of 768-dimensional ``float16`` embeddings, or of ``uint32`` token ids for ``tokens``, null where unmatched), and a ``bool`` per flag survey. Within an image cell the image tokens come first, then the survey's catalogue values; a spectrum cell leads with its normalisation token. Both are uncompressed Arrow IPC files that readers memory-map.

.. autofunction:: app.config.store_schema
   :no-index:

``probabilities`` is an uncompressed Arrow IPC file of AION's predictions at every slot. Image tokens keep their 64 most probable codes with their probabilities, the mass beyond the first 16 and the mass beyond all 64; search reads each galaxy's 16 most probable and spreads the rest evenly over the other codes, but the query galaxy's full 64; catalogue values and the redshift keep every probability; spectrum tokens keep 256 coefficients on a per-survey PCA basis, quantised to 8 bits as ``offset + code × step``.

.. autodata:: app.config.PREDICTIONS
   :no-index:
   :no-value:

``probability_basis`` is an ``np.savez`` of each spectrum survey's mean (``float64``, 1024) and top 256 eigenvectors (``float64``, 256 × 1024) of the spectrum token probabilities; a token's distribution is ``mean + coefficients @ directions``. ``generate_predictions`` fits the coefficients against it in ``float64``, and every later prediction must too; search decodes them in ``float32``.

``token_gram_<mode>`` and ``token_gram_<mode>_anywhere`` are ``np.save`` files of a galaxies × galaxies ``float32`` :term:`Gram matrix`: one for each mode at the same place, and one for each image and spectrum mode at any position. Entry :math:`(q, g)` is the mode score :math:`T_m(g)` of :doc:`/methods/similarity-search` with every slot of the mode selected on :math:`q` and every galaxy read as search reads the query galaxy, so a same-place matrix is symmetric up to rounding. The products run in TF32 with float32 accumulation. The app does not read them.

``token_embedding_<mode>`` and ``token_embedding_<mode>_anywhere`` are ``np.savez`` files of :term:`token embeddings <token embedding>`, one per Gram matrix :math:`K`. With :math:`K_c` the matrix less its row and column means, and :math:`s` and :math:`V` its top 1,024 singular values and right singular vectors: ``queries`` is :math:`K_c V s^{-1/2}` and ``galleries`` is :math:`V s^{1/2}`, galaxies × 1,024 ``float32`` each; ``offsets`` holds :math:`K`'s column means and ``self_scores`` its diagonal :math:`K(g, g)`, in ``float32``; ``singular_values`` holds all of :math:`K_c`'s in ``float64``, as square roots of :math:`K_c^\top K_c`'s eigenvalues, so values below about :math:`10^{-8}` of the largest are rounding error. ``queries[q] @ galleries.T + offsets`` ranks the galaxies as row :math:`q` of :math:`K` does, exactly at full rank. A new query's row :math:`k` of scores against every galaxy has the query vector :math:`(k - \bar k - o + \bar o)\, V s^{-1/2}`, with :math:`o` the offsets; for a galaxy's own row this is its stored query vector.

``token_search_<mode>`` and ``token_search_<mode>_anywhere``, from the same job, are ``np.savez`` files of the same ``queries`` and ``galleries`` cut to their first 256 components, and ``offsets``. The app loads them at startup for :term:`whole-mode search`, 392 MB for 17,369 galaxies (11 × 17,369 × (2 × 256 + 1) × 4 bytes), and it does not start without them.

``aion_prediction_<mode>`` is an ``np.save`` file of every galaxy's :term:`AION prediction` in the mode, ``float32``, one row per galaxy, observed in the mode or not. An image AION prediction is AION's codec decoding of the expected latent at every cell, :math:`\sum_t p_{g,s}(t)\, z(t)` with the mass beyond the stored codes spread evenly over the rest of the vocabulary, 4 Legacy Survey or 5 HSC bands × 96 × 96 px in the survey's flux units: the job decodes every band in Legacy Survey units and multiplies HSC's by :math:`10^{1.8}`, the inverse of the codec's scaling from HSC's zero point of 27. A spectrum AION prediction is the decoding of the expected latent at every spectral window without the normalisation token, 8,704 values on the codec's grid from 3,500 Å in 0.8 Å steps: a shape, since the normalisation token is not predicted. A catalogue value's or the redshift's AION prediction is the median of its predicted distribution over the codec's bins, each at the value AION decodes it to, its upper edge. The open top bin takes its lower edge instead: AION decodes it to the largest value in the codec's sample of its training data (1.9 × 10⁹ nmgy for ``FLUX_Z``) or to no value. Bins past it decode to no value and are skipped. Decoding runs in IEEE float32.

``text_views`` is a directory of Parquet parts, ``part-<n>.parquet``, of 4,000 :term:`views <view>` each, drawn from the galaxies outside text search's held-out set: the 30% held out of the text maps and the prototype's 143, which leaves 12,058. A view has one of 26 types: the observation, the :term:`AION prediction` or a sample in each :term:`mode`, or one of 5 combinations of modes in a single input. Each part draws its (galaxy, view type) pairs from its own seed, weighted 1 per AION prediction, observation and combination, 4 per image or spectrum sample and 8 per catalogue sample. A sample draws every slot's token from AION's stored distribution, the mass beyond an image cell's stored codes spread evenly over the rest, and decodes it with the mode's codec. An image is turned by one of its 8 rotations and reflections; a spectrum is smoothed by a Gaussian of σ = 1, 2 or 3 px and plotted at 1,680 × 384 px over the survey's observed range, its flux axis from the 0.5th to the 99.9th percentile with 10% padding; catalogue values are written one ``COLUMN: value`` line each, shuffled, at 3, 4 or 5 significant figures. A combination draws each member as observation, aion_prediction or sample, the observation only where one exists. ``round`` numbers each ten parts. A rerun keeps the parts already written, so a stopped job resumes. ``view_type`` is the type's position in ``app.config.VIEW_TYPES``:

.. autodata:: app.config.TEXT_VIEWS
   :no-index:
   :no-value:

``text_views_held_out`` holds the views ``train_text_regression`` checks the regression against: one of every type available to each of 500 :term:`held out` galaxies, the prototype's 143 among them, drawn from a random stream of their own:

.. autodata:: app.config.TEXT_VIEWS_HELD_OUT
   :no-index:
   :no-value:

``text_bases`` is an ``np.savez`` of the coordinates ``train_text_regression`` fits on the first round of ``text_views``. Per view type, ``mean`` is the mean of its embeddings (26 × 1,024), and a 128-component PCA of its mean-subtracted unit embeddings gives ``components`` (26 × 128 × 1,024), ``centre`` (26 × 1,024) and ``scale`` (26 × 128), each component's standard deviation, raised to at least a tenth of the type's median so that no target is divided by a near-zero spread; a type with fewer than 128 views in the round takes its PCA from its mode's three types together. ``offset`` and ``projection`` whiten ``joint_embedding`` to 256 components over the training galaxies. ``text_regression`` is a ``torch.save`` of the regression's weights: a network that predicts a view's 128 coefficients from its type and the whitened joint embedding, trained by squared error. In training, each of the joint embedding's 11 blocks is dropped with probability 0.3, and all of them with probability 0.1. It first runs ``generate_text_views`` for 200 parts and the held-out views, then trains on all 800,000 stored views for 111,000 steps of 512 views each, drawn so that every view type a galaxy has is as likely as any other, and keeps an exponential moving average of its weights, which it stores at the end. Every 6,000 steps it logs the training loss and the :term:`identification` of the held-out views, scoring each by its dot product with a galaxy's prediction for the view's type; neither decides when to stop. Its final report, which adds the identification of the prototype's galaxies and with each mode's blocks dropped or alone, goes to Weights & Biases and the job's log.

``text_vectors`` is an ``np.save`` file, ``float32``, galaxies × 7 modes × 1,024: for each galaxy and mode, the sum over view types :math:`k` that need the mode of :math:`\bar e_{g,k} / (26\,|M_k|)` (:doc:`/methods/text-search`), with :math:`\bar e_{g,k}` the regression's prediction with every block of the joint embedding present: its coefficients times ``scale``, through ``components``, plus ``centre``. Summed over modes, a galaxy's row is its mean prediction; the app sums it in batches on the first text search and reads the matches' rows for their contributions. ``text_query_mean`` is an ``np.save`` file of the mean WeMM-Embedding embedding of the 10,080 generic queries, each embedded after the search prefix as every query is (``float32``, 1,024), which every query has subtracted before scoring. WeMM-Embedding loads with the tokenizer normalizer its repository ships, which current ``transformers`` drops, and on a GPU in bfloat16 with flash attention from the kernels hub. The server sends each text query to ``QueryEmbedder.embed``, a method on whichever of an L4, A10G or L40S GPU is free first that stops after five idle minutes: on CPUs without AVX-512, a 9B query took about 6 s in bfloat16. It starts from a GPU memory snapshot taken after it loads the model from the volume, fetches the flash-attention kernel from the kernels hub and embeds one query, which compiles flash-linear-attention's Triton kernels. The starts that take Modal's snapshots, the first few on each GPU type after a deploy, do all of this in full.

``data_gram_<mode>`` and ``data_gram_<mode>_anywhere`` are ``np.save`` files of :term:`against-data scores <against-data score>`, ``float32``, one row per galaxy and one column per galaxy observed in the mode with data on the aion_prediction's pixels, in the order of ``observed_reference``'s ``<mode>_galleries``. With :math:`\mu_q` the query's :term:`AION prediction`, :math:`\mu_0` the median of all galaxies' AION predictions, and :math:`x_g` and :math:`w_g` the gallery galaxy's data and inverse variances on the AION prediction's pixels (the central 96 × 96 px of an image; for a spectrum, the codec's grid, through the transpose of the linear interpolation to each observed wavelength; for catalogue values and the redshift, the errors of ``aion_prediction_quality``):

.. math::

   S(q, g) = \langle \mu_q - \mu_0,\; w_g x_g \rangle - \tfrac12 \langle \mu_q^2 - \mu_0^2,\; w_g \rangle = S_\text{sat}(g) - \tfrac12 \chi^2(x_g; \mu_q)

with :math:`\chi^2(x_g; \mu) = \langle (x_g - \mu)^2,\; w_g \rangle` and :math:`S_\text{sat}(g) = \tfrac12 \chi^2(x_g; \mu_0)` the gallery galaxy's :term:`saturation`. A spectrum AION prediction takes its best amplitude, :math:`S(q, g) = \tfrac12 \left[ \langle \mu_q, w_g x_g \rangle^2 / \langle \mu_q^2, w_g \rangle - \langle \mu_0, w_g x_g \rangle^2 / \langle \mu_0^2, w_g \rangle \right]`. At any position, each of the query's :term:`slots <slot>` (a 4 × 4 px cell in every band, or one of the 32-pixel windows the survey covers) takes its best score over the gallery galaxy's slots with data, each against :math:`\mu_0` at that slot, and the slot scores add, so :math:`S` can exceed :math:`S_\text{sat}`; a spectrum's data are first divided by the best amplitude of the galaxy's own AION prediction. Same-place scores are computed in float64, any-position ones in IEEE float32.

``aion_prediction_gram_<mode>`` and ``aion_prediction_gram_<mode>_anywhere`` are ``np.save`` files of galaxies × galaxies ``float32`` :term:`AION prediction Grams <AION prediction Gram>`. Each AION prediction is scaled to the mode's :term:`typical depth`, :math:`a_g = \mu_g \sqrt{\bar w}`, with :math:`\bar w` the median of :math:`w_g` over the gallery galaxies; for a spectrum, :math:`w_g` is in units of the galaxy's own AION prediction's amplitude and :math:`a_g` is scaled to unit length. Entry :math:`(q, g)` is :math:`-\tfrac12 \lVert a_q - a_g \rVert^2`, or at any position the sum over the query's slots of :math:`-\tfrac12 \lVert a_{q,s} - a_{g,s'} \rVert^2` at the gallery galaxy's nearest slot :math:`s'`. As for the scores, same-place entries are computed in float64 and any-position ones in IEEE float32.

``observed_reference`` is an ``np.savez`` holding, for each mode, ``<mode>_reference`` (:math:`\mu_0`), ``<mode>_galleries`` (the galaxies observed in the mode with data on the aion_prediction's pixels), ``<mode>_saturation`` (each one's :term:`saturation`), ``<mode>_depth`` (:math:`\sqrt{\bar w}`) and, for spectra, ``<mode>_amplitudes`` (each galaxy's own AION prediction's best amplitude on its data).

``aion_prediction_embedding_<mode>`` and ``aion_prediction_embedding_<mode>_anywhere`` are :term:`AION prediction embeddings <AION prediction embedding>`, ``np.savez`` files with the parts of a token embedding, decomposed from the aion_prediction Grams, and ``widths``, each galaxy's :term:`predictive width` in the mode: for an image or spectrum the root mean square over slots of the variance of AION's predicted codec latent, summed over its dimensions; for catalogue values and the redshift the root mean square over values of half the width of the predicted central 68%, times :math:`\sqrt{\bar w}`.

``joint_embedding`` is an ``np.save`` file of every galaxy's :term:`joint embedding` from its token embeddings, galaxies × 2,816 ``float32``: the 11 blocks' 256 columns each, in ``GRAM_NAMES`` order, so any block can be taken alone.

``properties`` is a Parquet file of every galaxy's redshift, stellar mass (``mass``, :math:`\log_{10} M_\odot`), star formation rate (``sfr``, :math:`\log_{10} M_\odot\,\text{yr}^{-1}`), specific star formation rate (``ssfr``, :math:`\log_{10} \text{yr}^{-1}`), :term:`half-light radius` (``radius``, arcseconds), r magnitude (``r_magnitude``), :term:`axis ratio` (``axis_ratio``), Hα :term:`equivalent width` (``halpha_ew``, Å) and :term:`Dn4000` (``dn4000``), each measured where it has a value and predicted for every galaxy, with the prediction's width. The measured values are:

- the spectroscopic redshift where usable, DESI's before SDSS's;
- :term:`PROVABGS`'s mass and star formation rate;
- from the Legacy Survey's model of the galaxy: ``SHAPE_R``; :math:`22.5 - 2.5 \log_{10}` of the r flux in nanomaggies, null where the flux is not positive; and the axis ratio, null where both ellipticity components are 0, as in models round by construction;
- from the spectrum of the survey whose redshift is used, moved to the rest frame: Hα's equivalent width over 6,554.6 to 6,574.6 Å, against the mean of the median fluxes over 6,490 to 6,530 Å and 6,600 to 6,640 Å, positive for emission; and the Dn4000, the mean of :math:`f_\lambda \lambda^2` over 4,000 to 4,100 Å divided by its mean over 3,850 to 3,950 Å. Each band must lie inside the spectrum's unmasked pixels with at least 5 of them, and the continuum and both of the Dn4000's means must exceed three times their errors. So Hα is measured up to redshift 9,824 / 6,640 − 1 = 0.48 in DESI and 9,221.5 / 6,640 − 1 = 0.39 in SDSS.

The redshift's prediction is the median of AION's predicted distribution, and its width half the range from the distribution's 16th to its 84th percentile. Both spread each bin's probability evenly over the redshifts its token covers, from the bin below up to its own value: a distribution in one bin, 0.0059 wide, predicts the bin's centre with a width of 0.0020, where the redshift :term:`AION prediction` takes the bin's top. A galaxy with a measured redshift read it, so its prediction is close to it; ``redshift_quality`` measures predictions without it. Likewise the joint embedding reads the Legacy Survey catalogue values, so the radius, r magnitude and axis ratio predictions are close to the measured ones. The others come from ridge regression on the joint embedding, its penalty chosen by leave-one-out, fitted to the galaxies with a measured value. Those galaxies' own predictions come from 5-fold cross-validation, so none is fitted to its own value, and the RMS error of those predictions is the width, the same for every galaxy. PROVABGS fits DESI spectra, so its widths are measured only on galaxies DESI observed, and the spectral ones only on galaxies with a spectrum. ``g_r`` is the observed Legacy Survey colour, :math:`-2.5 \log_{10}` of the g to r flux ratio, null where either flux is not positive. ``/galaxies/property`` serves it.

``patterns`` is an Arrow IPC file of each galaxy's ``galaxy`` and ``colours``, 24 × 24 × 3 ``uint8``. ``generate_patterns`` fits scikit-learn's ``PCA`` with 3 components to 100,000 Legacy Survey image patch embeddings from ``encoded`` (768 numbers each), drawn at random with ``SEED`` from every galaxy's 576, then projects every patch, 256 galaxies per batch. Each component, clipped at its 2nd and 98th percentiles over all patches and scaled to 0 to 255, is one of red, green and blue. ``/patterns`` serves the file.

.. autodata:: app.config.PROPERTIES
   :no-index:
   :no-value:

``mean_points`` (one row per galaxy) and ``full_points`` (one row per embedding, written survey by survey) are Parquet, and so are ``same_place_points`` and ``any_position_points``, one row per galaxy placed by :term:`whole-mode search` over every mode (:doc:`/methods/map`):

.. autodata:: app.config.POINTS
   :no-index:
   :no-value:

``pairs_image`` and ``pairs_spectrum`` are Parquet, in galaxy order. ``pairs_image`` holds, for every galaxy, the mean of AION's Legacy Survey image embeddings and WeMM-Embedding's embedding of the colour image; ``pairs_spectrum`` holds, for every galaxy with a DESI spectrum, the mean of AION's DESI spectrum embeddings and WeMM-Embedding's embedding of the spectrum plot:

.. autodata:: app.config.PAIRS
   :no-index:
   :no-value:

``parametric_umap`` is a ``torch.save`` file of the projector's weights.

HTTP API
--------

Every endpoint needs the password in the ``access-password`` Modal secret's ``ACCESS_PASSWORD``, sent as ``Authorization: Bearer <password>`` or in the ``password`` cookie, and otherwise answers ``401``. ``POST /login`` takes the bearer header and sets that cookie for 30 days, ``HttpOnly``, ``Secure`` and ``SameSite=Lax``. ``/docs``, ``/redoc`` and ``/openapi.json`` are public.

Every other endpoint is ``GET``; ``/projections/{projection}`` and ``/galaxies/property`` also answer ``HEAD``. A request whose ``If-None-Match`` matches the response's ETag gets ``304 Not Modified``. Artifact files carry Starlette's ETag, from size and modification time; every other response's ETag is an MD5 of its body. Every response carries ``Cache-Control: no-cache``.

The first Same place Population map (``/saliency``) on a galaxy builds its agreement matrix, the log overlap of every galaxy at every slot, and the server keeps the 8 most recently used. An Any position map builds no matrix: it computes the rows of its sample of galaxies (:doc:`/methods/query-maps`), and the server keeps them for the 8 most recently used pairs of galaxy and Any position modes. These computations, the matrix builds and the sums of a new Any position search run one at a time.

.. openapi:: ../../web/openapi.json

Tests and CI
------------

Tests run on a synthetic artifact tree and dataset that ``scripts/fixture.py`` builds into ``.fixtures`` in the production schemas, with AION and its codecs at random weights; no test downloads weights. ``/galaxy/{galaxy}/image/{survey}``, ``/galaxy/{galaxy}/spectrum/{survey}``, WeMM-Embedding, ``generate_pairs`` and the observed views of ``generate_text_views`` are untested, ``generate_token_grams`` is tested on the CPU in float32 rather than TF32, ``generate_maps`` is tested for its neighbours but not its UMAP layout, which needs more galaxies than UMAP's 15 neighbours, and the web app has no tests.

Continuous integration lints and tests every pull request, checks that ``web/openapi.json`` is current and builds these docs with warnings as errors. On ``main``, once lint, tests and the ``openapi.json`` check pass, it deploys to Modal's ``main`` environment. Running the workflow by hand on any branch deploys that branch to the ``dev`` environment, which has its own app, volume and secrets. Vercel builds every branch: production builds from ``main`` and calls the ``main`` app, and preview builds from every other branch call the ``dev`` app, whose URL ``PUBLIC_API_URL`` holds in each Vercel environment. ``web/vercel.ts`` reads it at build time to rewrite ``/api`` to that app, so the browser keeps one origin and sends the cookie with every request, including DuckDB's. Each Modal environment needs its own ``access-password`` secret. The ``dev`` app runs whichever branch last ran the workflow, so previews share it.
