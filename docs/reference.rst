Reference
=========

How it fits together
--------------------

The build pipeline and the app share nothing but files. Seven Modal jobs write artifacts to a volume mounted at ``/cache``, under ``$ALPHAUNIVERSE_CACHE/<author>/<name>/<revision>/``, so changing ``DATASET_REVISION`` switches trees rather than overwriting one. The FastAPI app in ``app/main.py`` reads them in one Modal container; the SvelteKit frontend, deployed on Vercel, calls it. ``main`` deploys to Modal through ``.github/workflows/ci-cd.yml``.

========================= ============================================= =======================================
Job                       Reads                                         Writes
========================= ============================================= =======================================
generate_embeddings       the dataset                                   encoded, codebook, tokens
generate_predictions      tokens                                        probabilities, probability_basis
generate_token_grams      probabilities, probability_basis              token_gram_<mode>,
                                                                        token_gram_<mode>_anywhere
generate_projections      the dataset, encoded                          mean_points, full_points,
                                                                        parametric_umap
generate_pairs            the dataset, encoded                          pairs_<modality>
generate_alignment        pairs_<modality>                              alignment
generate_aion_gemma_space pairs_<modality>, alignment                   aion_gemma_space
========================= ============================================= =======================================

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

``probability_basis`` is an ``np.savez`` of each spectrum survey's mean (``float64``, 1024) and top 256 eigenvectors (``float64``, 256 × 1024) of the spectrum token probabilities; a token's distribution is ``mean + coefficients @ directions``.

``token_gram_<mode>`` and ``token_gram_<mode>_anywhere`` are ``np.save`` files of a galaxies × galaxies ``float32`` matrix: one for each mode at the same place, and one for each image and spectrum mode at any position. Entry :math:`(q, g)` is the mode score :math:`T_m(g)` of :doc:`methods/similarity-search` with every slot of the mode selected on :math:`q` and every galaxy read as search reads the query galaxy, so a same-place matrix is symmetric up to rounding. The products run in TF32 with float32 accumulation. The app does not read them.

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

Tests run on a synthetic artifact tree that ``scripts/fixture.py`` builds in the production schemas, with AION and its codecs at random weights; no test downloads weights. ``/galaxy/{galaxy}/image/{survey}``, ``/galaxy/{galaxy}/spectrum/{survey}``, EmbeddingGemma, ``generate_pairs``, ``generate_alignment`` and ``generate_aion_gemma_space`` are untested, ``generate_token_grams`` is tested on the CPU in float32 rather than TF32, and the frontend has no tests.

Continuous integration lints and tests every pull request, checks that ``frontend/openapi.json`` is current and builds these docs with warnings as errors. On ``main``, once lint, tests and the ``openapi.json`` check pass, it deploys to Modal's default environment. Running the workflow by hand on any branch deploys that branch to the ``dev`` environment, which has its own app, volume and secrets. Vercel builds every branch: production builds from ``main`` and calls the default app, and preview builds from every other branch call the ``dev`` app, whose URL ``PUBLIC_API_URL`` holds in each Vercel environment. The ``dev`` app runs whichever branch last ran the workflow, so previews share it.
