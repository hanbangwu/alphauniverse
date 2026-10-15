Benchmarks
==========

Every benchmark runs on Modal against the artifacts on the production volume, and writes a report to ``docs/benchmarks/<script>.json``. The charts and tables below are drawn from those reports when the documentation is built, so each shows exactly one run, named by its commit and date, and a newly committed report updates its section. Figures compare only within one run, never across runs or machines. :doc:`how-to` explains how to run them.

Projection quality
------------------

How faithfully does the map keep neighbours? ``projection_quality`` takes 10,000 embeddings held out of the projector's training, projects them, and compares each embedding's :math:`k` nearest neighbours by cosine distance in 768 dimensions, :math:`N_k(i)`, with its :math:`k` nearest on the map, :math:`\hat N_k(i)`, for :math:`k = 15` and :math:`100`:

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

Can a sentence find the right galaxies? ``text_search_quality`` uses the 30% of galaxies held out of the text maps' training, and six queries whose answers are PROVABGS cuts:

======================================================= ==============================
Query                                                   Answer
======================================================= ==============================
A massive galaxy, stellar mass above 10^11 solar masses :math:`\log M_* > 11`
A low-mass dwarf galaxy                                 :math:`\log M_* < 9.5`
A star-forming galaxy                                   :math:`\log \mathrm{sSFR} > -10`
A quiescent galaxy with no ongoing star formation       :math:`\log \mathrm{sSFR} < -11`
A distant galaxy at redshift above 0.5                  :math:`z > 0.5`
A nearby galaxy at redshift below 0.1                   :math:`z < 0.1`
======================================================= ==============================

with :math:`M_*` in solar masses and sSFR in yr\ :sup:`-1`. Each query ranks the galaxies that have that PROVABGS property in three spaces per modality, image or spectrum: EmbeddingGemma's own embeddings, and AION's embeddings carried into EmbeddingGemma's space by a linear map or by the network the site uses (MLP). The spectrum spaces rank only galaxies with a DESI spectrum. *Combined MLP* ranks every galaxy by the higher of its image and spectrum rows' scores, as the site ranks them. Each is ranked raw and *centred*, with the mean document and mean query subtracted first.

The *base rate* is the fraction of galaxies that satisfy the cut, the score of a random ranking. *Average precision* averages, over every correct galaxy, the fraction correct among the galaxies ranked at or above it; *precision@k* is the fraction correct in the top :math:`k`. A useful ranking beats the base rate.

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

How closely does AION's predicted observation match a galaxy's own data? ``template_quality`` compares every galaxy's :term:`template` in each mode it was observed in with its data by the reduced :math:`\chi^2`: images over the central 96 × 96 px with each pixel's inverse variance; spectra on their own wavelength grid with a free amplitude; catalogue values and redshift with the dataset's errors where it has them (HSC magnitudes, redshift) and otherwise the population's robust scatter, 1.4826 times the median absolute deviation. A galaxy with nothing to compare in a mode, no valid pixel in the crop, fewer than two in a spectrum or no catalogue value with an error, is left out. Beside the template it scores two references: the floor, the codec's decoding of the galaxy's own tokens, and the null, the mean template over all galaxies. *Recovered* is :math:`(\chi^2_\text{null} - \chi^2_\text{template}) / (\chi^2_\text{null} - \chi^2_\text{floor})`: 1 when the template fits as well as the galaxy's own tokens, 0 when it fits no better than the mean template. The normalised residuals :math:`(x - \mu)\sqrt{w}` are averaged by radius for images and by rest wavelength for spectra with a usable redshift.

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

For each mode and for the two full-object scores of ``token_embedding_quality``, it reports the share of galaxies that rank themselves first (R@1) and in their top 10 (R@10), and their median rank, a galaxy tied with others taking their mean rank. A Gram matrix's row need not peak at its own galaxy, so the set with all the data is each score's ceiling.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import retrieval_quality

   show(retrieval_quality())

Property probes
---------------

How much of a galaxy's physics do the stored vectors hold? ``probe_quality`` predicts :term:`PROVABGS` stellar mass, :math:`\log_{10}` star formation rate and redshift for the galaxies that have them, from two kinds of features: AION's mean embeddings, and each mode's first 256 stored gallery vectors, scaled to unit root-mean-square length about their mean and concatenated over the modes. It fits ridge regression, with its penalty chosen by leave-one-out cross-validation, and the mean of the 10 nearest neighbours, and reports :math:`R^2` and the RMS error under 5-fold cross-validation, with every mode and without the two spectrum modes. Redshift is predicted without the redshift mode's vectors, but every feature comes from an encoding that read the galaxy's redshift where it has one, so it measures how much of that input the features keep.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import probe_quality

   show(probe_quality())

Search speed
------------

In a container built like the server, ``search_performance`` times the checked-out commit and, when its code differs, the stored report's fastest version, alternating them so neither always runs first. Every figure is from a running server: the median time of each stage of a search, building a galaxy's agreement matrix and the Population map from it on 4 galaxies, whole searches with 4 image cells at 32, 64 and 128 matches, and an any-position search over 4 image cells with its first and repeated Population map.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import search_performance

   show(search_performance())

Request speed
-------------

``backend_performance`` starts a temporary copy of the server and times requests to it from a separate container, including Modal's network but not the visitor's. For ``/meta``, a similarity search, a text search, a galaxy's image, its spectrum and its catalogue table, it times the first request of each kind to a freshly started server (cold, paying for what that request loads first) and then 30 more to the running server (warm).

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import backend_performance

   show(backend_performance())
