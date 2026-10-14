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

Can a sentence find the right galaxies? ``text_search_quality`` uses the 30% of galaxies held out of the text map's training, and six queries whose answers are PROVABGS cuts:

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

with :math:`M_*` in solar masses and sSFR in yr\ :sup:`-1`. Each query ranks the galaxies that have that PROVABGS property in three spaces: EmbeddingGemma's own image embeddings, and AION's embeddings carried into EmbeddingGemma's space by a linear map or by the network the site uses (MLP). Each is ranked raw and *centred*, with the mean document and mean query subtracted first.

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

Beside it are the same shares for positive-only classical MDS of each matrix's symmetric part, ranked by :math:`-\tfrac12 \lVert E_q - E_g \rVert^2 + \tfrac12 K(g, g)`; the share of that symmetric part's spectrum, :math:`\sum |\lambda|`, in negative eigenvalues, which no Euclidean embedding holds; the share of the squared singular values in the first :math:`k`; and, at any position, the share of the matrix's variance about its mean in its antisymmetric part, with the share of the exact top 10 that its symmetric part finds.

.. bokeh-plot::
   :source-position: none

   from bokeh.plotting import show

   from benchmark_plots import token_embedding_quality

   show(token_embedding_quality())

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
