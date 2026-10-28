Benchmarks
==========

Every benchmark runs on Modal against the artifacts on the production volume, and writes a report to ``docs/benchmarks/<script>.json``. The charts and tables below are drawn from those reports when the documentation is built, so each shows exactly one run, named by its commit and date, and a newly committed report updates its section. Figures compare only within one run, never across runs or machines. :doc:`how-to` explains how to run them.

Projection quality
------------------

How faithfully does the :term:`parametric UMAP` projection behind ``mean_points`` and ``full_points`` keep neighbours? ``projection_quality`` takes 10,000 embeddings held out of the projector's training, projects them, and compares each embedding's :math:`k` nearest neighbours by cosine distance in 768 dimensions, :math:`N_k(i)`, with its :math:`k` nearest on the map, :math:`\hat N_k(i)`, for :math:`k = 15` and :math:`100`:

.. math::

   \text{preservation} = \frac{1}{n} \sum_i \frac{|N_k(i) \cap \hat N_k(i)|}{k},
   \qquad
   \text{trustworthiness} = 1 - \frac{2}{n k (2n - 3k - 1)}
   \sum_i \sum_{j \in \hat N_k(i)} \max\bigl(0,\, r(i, j) - k\bigr),

where :math:`r(i, j)` is :math:`j`'s rank among :math:`i`'s neighbours in 768 dimensions. Preservation is 1 when the map keeps every neighbour; random positions would keep :math:`k / (n - 1)`. Trustworthiness penalises points that look close on the map but are far apart in 768 dimensions; random positions score near 0.5.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import projection_quality

   show(projection_quality())

Text search quality
-------------------

Can a sentence find the right galaxies? ``text_search_quality`` ranks the 30% of galaxies :term:`held out` of the text maps' training with six queries whose answers are cuts on measured values, the :term:`PROVABGS` stellar mass and specific star formation rate and the spectroscopic redshift:

================================================= ================================
Query                                             Answer
================================================= ================================
a massive galaxy                                  :math:`\log M_* > 11`
a dwarf galaxy                                    :math:`\log M_* < 9.5`
a star-forming galaxy                             :math:`\log \mathrm{sSFR} > -10`
a quiescent galaxy with no ongoing star formation :math:`\log \mathrm{sSFR} < -11`
a distant galaxy at high redshift                 :math:`z > 0.5`
a nearby galaxy at low redshift                   :math:`z < 0.1`
================================================= ================================

with :math:`M_*` in solar masses and sSFR in yr\ :sup:`-1`. Each galaxy is a set of points in :term:`EmbeddingGemma`'s space, one per modality it has: EmbeddingGemma's embeddings of its colour image and DESI spectrum plot (*gemma*), or AION's embeddings carried there by the site's networks (*maps*). A query :math:`t` scores galaxy :math:`g` in two ways. *Raw* is the highest cosine between :math:`t` and the galaxy's points, as the site ranks now. *Likelihood* is

.. math::

   s(g \mid t) = \log \frac{1}{n_g} \sum_{i=1}^{n_g} \exp\left(\kappa \cos(\tilde t, \tilde e_{g,i})\right),
   \qquad \kappa = 15,

over the galaxy's :math:`n_g` points :math:`e_{g,i}`, where :math:`\tilde e_{g,i}` is a point less the mean of its modality's points over all galaxies, and :math:`\tilde t` is the query less the mean of 10,704 generic queries: every distinct combination of four framings, 14 adjectives, 16 kinds of galaxy and 12 endings, such as *an image of a dusty spiral galaxy with a dust lane*.

The *base rate* is the fraction of galaxies that satisfy the cut, the score of a random ranking. *Average precision* averages, over every correct galaxy, the fraction correct among the galaxies ranked at or above it; its 95% interval spans the middle 95% of its values over 1,000 resamples of the galaxies. *precision@k* is the fraction correct in the top :math:`k`. A useful ranking beats the base rate.

A query's :term:`breadth` is :math:`\exp(H)/N`, with :math:`H` the entropy of :math:`p(g \mid t) \propto \exp s(g \mid t)` over the :math:`N` galaxies, at :math:`\kappa` = 15 and 20. It is measured along three ladders, each query more specific than the last: from *a galaxy* to *a red barred spiral galaxy with a ring*, from *a galaxy* to *a giant round red elliptical galaxy with a bright core*, and from *a galaxy spectrum* to *a spectrum with strong H-alpha and [OIII] emission lines at redshift 0.1*.

A modality's *share* of the top 10 or top 100 is, under *raw*, the fraction of those galaxies whose highest cosine is that modality's, and under *likelihood*, the mean over them of :math:`\sum_{i \in \text{modality}} \exp(\kappa \cos(\tilde t, \tilde e_{g,i})) \big/ \sum_i \exp(\kappa \cos(\tilde t, \tilde e_{g,i}))`.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import text_search_quality

   show(text_search_quality())

Similarity search quality
-------------------------

Does similarity search find galaxies that are physically alike? ``search_quality`` samples galaxies with a spectrum and a redshift, searches from each with four kinds of selection (16 consecutive spectrum tokens, every observed spectrum token, four Legacy Survey catalogue values, four HSC catalogue values) and reports, as a mean over queries with a 95% bootstrap interval:

- **redshift**: the median :math:`|\Delta z| / (1 + z)` between the query and its top 10 matches; lower is better. Its redshift is DESI's, else SDSS's, without the flag or limit AION's input uses.
- **identity**: the fraction of queries whose own galaxy, with its spectra and redshift hidden and predicted again, ranks in the top 10; higher is better.
- **availability** and **evidence**: each sampled galaxy is shown either with its spectrum or with it hidden, at random; these are how much more often a galaxy shown with its spectrum lands in the top 32 than its share, for galaxies more than 0.01 in :math:`|\Delta z| / (1 + z)` from the query (availability) and within it (evidence). Availability near 0 means having a spectrum alone does not lift a galaxy; positive evidence means a spectrum helps real matches.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import search_quality

   show(search_quality())

Prediction storage
------------------

How much does compressing AION's predictions change search results? ``compression_quality`` predicts a sample of galaxies exactly, stores them as the site does (:doc:`reference` describes the current format), and scores the same queries of several sizes on both. For each mode and selection size it reports the stored bytes per galaxy and, over queries, the median and 5th-percentile Spearman correlation with the exact ranking, the median overlap of the top 10, and the identity and redshift checks of ``search_quality`` under both. The figures show the format stored at the report's commit.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import compression_quality

   show(compression_quality())

Token embeddings
----------------

How much of a full-object search do the :term:`token embeddings <token embedding>` keep? ``token_embedding_quality`` ranks every galaxy against every other galaxy as a query, once from each mode's :term:`Gram matrix` and once from its stored vectors cut to their first :math:`k` components, and reports the share of the exact top 10 and top 100 that the vectors find. It does this for each matrix and for two full-object scores, each mode standardised per query as search combines modes: the seven same-place matrices, and the four any-position matrices with the catalogue values and redshift.

Beside it are the same shares for positive-only :term:`classical MDS` of each matrix's double-centred symmetric part, which keeps the components with positive eigenvalues and ranks by :math:`-\tfrac12 \lVert E_q - E_g \rVert^2 + \tfrac12 K(g, g)`, with :math:`E_g` galaxy :math:`g`'s coordinates; the share of that part's spectrum, :math:`\sum |\lambda|`, in negative eigenvalues, which no Euclidean embedding holds; the share of the squared singular values in the first :math:`k`; and, at any position, the share of the matrix's variance about its mean in its antisymmetric part, with the share of the exact top 10 that its symmetric part finds.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import token_embedding_quality

   show(token_embedding_quality())

Templates
---------

How closely does AION's predicted observation match a galaxy's own data? ``template_quality`` compares every galaxy's :term:`template` in each mode it was observed in with its data by the reduced :math:`\chi^2`: images over the central 96 × 96 px with each pixel's inverse variance; spectra on their own wavelength grid with a free amplitude; catalogue values and redshift with the dataset's errors where it has them (HSC magnitudes, redshift) and otherwise the population's robust scatter, 1.4826 times the median absolute deviation. A galaxy with nothing to compare in a mode, no valid pixel in the crop, fewer than two in a spectrum or no catalogue value with an error, is left out. Beside the template it scores two references: the floor, the codec's decoding of the galaxy's own tokens, and the null, the mean template over all galaxies. *Recovered* is :math:`(\chi^2_\text{null} - \chi^2_\text{template}) / (\chi^2_\text{null} - \chi^2_\text{floor})`: 1 when the template fits as well as the galaxy's own tokens, 0 when it fits no better than the mean template. The normalised residuals :math:`(x - \mu)\sqrt{w}` are averaged by radius for images and by rest wavelength for spectra with a usable redshift. Spectra are also split into quartiles of their median S/N per pixel, :math:`x\sqrt{w}` over the valid pixels: a faint spectrum leaves the null little room above the floor, so its recovered share is small whatever the template.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import template_quality

   show(template_quality())

Template embeddings
-------------------

How much of exact against-data search do the template embeddings keep? ``template_embedding_quality`` takes every galaxy as a query and the galaxies observed in each mode with data as the gallery, leaving the query out, and compares the top 10 and top 100 with four rankings: the :term:`against-data score` :math:`S`; :math:`S / S_\text{sat}`, the share of each gallery galaxy's :term:`saturation` the query's template reaches; :math:`S - S_\text{sat} = -\tfrac12 \chi^2`, the fit of the query's template to each gallery galaxy's data; and the :term:`template Gram`, which stands in for :math:`S - S_\text{sat}`: it replaces each gallery galaxy's data by its own template and its errors by the :term:`typical depth`, and compares spectra by shape alone. The template Gram's agreement with :math:`S - S_\text{sat}` measures these replacements, and with :math:`S` and :math:`S / S_\text{sat}` also the change of ranking; the stored vectors' agreement with the template Gram measures the components left out. At any position only :math:`S` and the template Gram are compared, since :math:`S` there has no saturation to bound it. For each galaxy observed in the mode, it also reports the rank of the galaxy's own data in its own row under each exact ranking.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import template_embedding_quality

   show(template_embedding_quality())

Retrieval from partial data
---------------------------

Does a galaxy seen with less of its data still find itself? ``retrieval_quality`` encodes sampled galaxies again from part of their data, predicts every mode from that encoding, and ranks every galaxy, with all of its data, against it: by the encoding's row of each :term:`Gram matrix` and :term:`template Gram`, exactly and through the stored :term:`token embeddings <token embedding>` and :term:`template embeddings <template embedding>` cut to their first :math:`k` components, and by the :term:`cosine similarity` of :term:`mean embeddings <mean embedding>`. Each set holds up to 512 galaxies and keeps:

- all of a galaxy's data;
- the Legacy Survey image and values alone, or HSC's alone, of a galaxy with both;
- the DESI spectrum alone, or SDSS's alone, of a galaxy with both;
- everything but the DESI spectrum and the redshift, of a galaxy with a DESI spectrum.

For each mode and for the two full-object scores of ``token_embedding_quality``, it reports the share of galaxies that rank themselves first (R@1) and in their top 10 (R@10), and their median rank, a galaxy tied with others taking their mean rank. A Gram matrix's row need not peak at its own galaxy, so the set with all the data is each score's ceiling. It also ranks the other way, as a search from the galaxy with all its data would: the encoding takes its galaxy's place among the stored galaxies and is ranked by the galaxy's own row. This uses the exact rows only, since an encoding has no stored gallery vector.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import retrieval_quality

   show(retrieval_quality())

Redshift without spectra
------------------------

How well does AION predict a redshift from images and catalogue values alone? ``redshift_quality`` encodes every galaxy with a usable spectroscopic redshift again from its Legacy Survey and HSC data, with its spectra and redshift hidden, and predicts the redshift. It compares the median of the predicted distribution, with each bin's probability spread over the redshifts its token covers as ``properties`` takes it, with the measured value, DESI's where usable and otherwise SDSS's. For :math:`\Delta = (z_\text{AION} - z) / (1 + z)` it reports the bias, the median of :math:`\Delta`; the scatter, 1.4826 times its median absolute deviation; and the outlier rate, the share with :math:`|\Delta| > 0.15`. The redshifts ``properties`` predicts from the stored predictions, which read the spectra and redshift, are scored the same way as the reference, over all galaxies and by survey.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import redshift_quality

   show(redshift_quality())

Property probes
---------------

How much of a galaxy's physics do the stored vectors hold? ``probe_quality`` predicts :term:`PROVABGS` stellar mass, :math:`\log_{10}` star formation rate, :math:`\log_{10}` specific star formation rate and redshift for the galaxies that have them, from three kinds of features: AION's mean embeddings; the :term:`joint embedding`, each mode's first 256 stored gallery vectors, scaled to unit root-mean-square length about their mean and concatenated over the modes; and the first 1,408, 704 and 352 principal components of that concatenation over all galaxies. It fits ridge regression, with its penalty chosen by leave-one-out cross-validation, and the mean of the 10 nearest neighbours, and reports :math:`R^2` and the RMS error under 5-fold cross-validation, with every mode and without the two spectrum modes. Redshift is predicted without the redshift mode's vectors, but every feature comes from an encoding that read the galaxy's redshift where it has one, so it measures how much of that input the features keep. To show how few examples each needs, it also fits each set to 32, 128 and 512 galaxies and reports :math:`R^2` on a fixed fifth of the others, the mean and standard deviation over 20 draws, with the median ridge penalty and how often it sits at an end of the grid it is chosen from. The same draws fit each mode's vectors alone, and a stack of them: ridge on each mode, combined with non-negative weights fitted to the modes' cross-validated predictions; it reports the weights of a stack fitted to every galaxy.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import probe_quality

   show(probe_quality())

Neighbourhoods
--------------

Are a galaxy's nearest galaxies like it, and does one vector per galaxy show them? ``neighbour_quality`` asks three things of the stored :term:`token embeddings <token embedding>` and :term:`template embeddings <template embedding>`, cut to their first 256 components, of their :term:`joint embeddings <joint embedding>` over all modes, of the token one's first 1,408, 704 and 352 principal components, and of AION's :term:`mean embeddings <mean embedding>`:

* how far in :term:`PROVABGS` stellar mass, :math:`\log_{10}` star formation rate, :math:`\log_{10}` specific star formation rate and redshift a galaxy's 10 nearest are, against 10 at random: the galaxies with all four are both the queries and the gallery, ranked by full-object search at the same place, at any position and at the same place without the redshift mode, by Euclidean distance in each joint embedding and principal-component cut, and by the :term:`cosine similarity` of mean embeddings;
* how much of each mode's search top 10 the Euclidean top 10 of its gallery vectors keeps, and of full-object search each joint embedding's and principal-component cut's, for 2,000 queries against every galaxy;
* how much of each galaxy's 15 and 100 nearest, by full-object search at the same place in either space or by the cosine of mean embeddings, a map keeps nearest: a UMAP map built from each galaxy's 14 nearest by the token search, UMAP maps of the joint embeddings and the principal-component cuts, and the site's :term:`parametric UMAP` map of mean embeddings.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import neighbour_quality

   show(neighbour_quality())

Search speed
------------

In a container built like the server, ``search_performance`` times the checked-out commit and, when its code differs, the stored report's fastest version, alternating them so neither always runs first. Every figure is from a running server: the median time of each stage of a search, building a galaxy's agreement matrix and the Population map from it on 4 galaxies, whole searches with 4 image cells at 32, 64 and 128 matches, and an any-position search over 4 image cells with its first and repeated Population map. For whole-mode search it times loading the 256-component copies of the token embeddings, then searches over all seven modes at 32 matches, at the same place and at any position: the ranking, the maps, the missing-data marks and the whole search.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import search_performance

   show(search_performance())

Request speed
-------------

``backend_performance`` starts a temporary copy of the server and times requests to it from a separate container, including Modal's network but not the visitor's. For ``/meta``, a similarity search, a whole-mode search over all seven modes, a text search, a galaxy's image, its spectrum and its catalogue table, it times the first request of each kind to a freshly started server (cold, paying for what that request loads first) and then 30 more to the running server (warm).

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import backend_performance

   show(backend_performance())
