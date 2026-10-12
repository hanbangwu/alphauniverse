Benchmarks
==========

Every benchmark runs on Modal against the artifacts the live site serves, and writes a report to ``docs/benchmarks/<script>.json``. The charts and tables below are drawn from those reports when the documentation is built, so each shows exactly one run, named by its commit and date. Figures compare only within one run, never across runs or machines. :doc:`how-to` explains how to run them.

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

   import json
   from pathlib import Path

   from bokeh.layouts import column
   from bokeh.models import ColumnDataSource, DataTable, Div, FactorRange, TableColumn
   from bokeh.plotting import figure, show

   report = json.loads(Path("benchmarks/projection_quality.json").read_text())
   metrics = ("preservation", "trustworthiness")
   rows = [
       (str(neighbours), metric, values[metric])
       for neighbours, values in report["neighbours"].items()
       for metric in metrics
   ]
   source = ColumnDataSource(
       {
           "factor": [(neighbours, metric) for neighbours, metric, _ in rows],
           "neighbours": [neighbours for neighbours, _, _ in rows],
           "metric": [metric for _, metric, _ in rows],
           "value": [value for _, _, value in rows],
       }
   )
   chart = figure(
       x_range=FactorRange(*source.data["factor"]),
       y_range=(0, 1),
       height=300,
       title="Neighbours kept, by k",
       tooltips=[("k", "@neighbours"), ("metric", "@metric"), ("value", "@value")],
   )
   chart.vbar(x="factor", top="value", width=0.8, source=source)
   table = DataTable(
       source=source,
       columns=[TableColumn(field=name, title=name) for name in ("neighbours", "metric", "value")],
       index_position=None,
       height=140,
   )
   show(
       column(
           Div(
               text=f"Commit <code>{report['commit']}</code>, {report['date'][:10]}; "
               f"{report['size']:,} embeddings."
           ),
           chart,
           table,
       )
   )

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

   import json
   from pathlib import Path

   from bokeh.layouts import column
   from bokeh.models import ColumnDataSource, DataTable, Div, FactorRange, TableColumn
   from bokeh.plotting import figure, show

   report = json.loads(Path("benchmarks/text_search_quality.json").read_text())
   rows = [
       {"query": query, "space": space, "centring": centring} | figures
       for query, spaces in report["queries"].items()
       for space, centrings in spaces.items()
       for centring, figures in centrings.items()
   ]
   raw = [row for row in rows if row["centring"] == "raw"]
   chart_source = ColumnDataSource(
       {
           "factor": [(row["query"], row["space"]) for row in raw],
           "average_precision": [row["average_precision"] for row in raw],
           "base_rate": [row["base_rate"] for row in raw],
       }
   )
   chart = figure(
       y_range=FactorRange(*reversed(chart_source.data["factor"])),
       x_range=(0, 1),
       height=560,
       width=760,
       title="Average precision, raw (bars) against the base rate (ticks)",
       tooltips=[("average precision", "@average_precision"), ("base rate", "@base_rate")],
   )
   chart.hbar(y="factor", right="average_precision", height=0.8, source=chart_source)
   chart.scatter(x="base_rate", y="factor", marker="dash", angle=1.5708, size=18, color="black", source=chart_source)
   names = ("query", "space", "centring", "base_rate", "average_precision", "precision@10", "precision@100")
   table = DataTable(
       source=ColumnDataSource({name: [row[name] for row in rows] for name in names}),
       columns=[TableColumn(field=name, title=name) for name in names],
       index_position=None,
       width=760,
       height=400,
   )
   show(
       column(
           Div(
               text=f"Commit <code>{report['commit']}</code>, {report['date'][:10]}; "
               f"{report['galaxies']:,} held-out galaxies."
           ),
           chart,
           table,
       )
   )

Similarity search quality
-------------------------

Does similarity search find physically similar galaxies? ``search_quality`` samples galaxies with a DESI or SDSS spectrum and a redshift, and searches from each with four selections: 16 consecutive observed spectrum tokens, every observed spectrum token, four random Legacy Survey catalogue values, and, for galaxies with an HSC match, four random HSC catalogue values. It reports, as a mean over queries with a 95% bootstrap interval:

- **redshift**: the median :math:`|\Delta z| / (1 + z)` of the top 10 matches with a known redshift; lower is better.
- **identity**: each sampled galaxy is predicted again with its spectra and redshift hidden; the fraction of spectrum queries for which that hidden copy would rank in the top 10.
- **availability** and **evidence**: in 20 draws, half the sampled galaxies are swapped for their hidden copies. Among the top 32, the share that kept their spectrum, minus that share in the pool, over galaxies more than :math:`|\Delta z| / (1 + z) = 0.01` from the query (availability) and within it (evidence). Availability above 0 means merely having a spectrum lifts a galaxy; evidence above 0 means a spectrum helps the galaxies that do resemble the query.

No report from the current script is committed yet.

Similarity search speed
-----------------------

In a container built like the server, ``search_performance`` times the checked-out commit and, when its code differs, the stored report's fastest version, alternating them so neither always runs first. Every figure is from a running server:

- the median time of each stage of a search, and its share of the whole;
- building a galaxy's agreement matrix, and the Population map from it, on 4 galaxies;
- whole searches with 4 image cells selected, at 32, 64 and 128 matches;
- an any-position search over 4 image cells, and its Population map.

No report from the current script is committed yet.

Storage quality
---------------

AION predicts a probability for every code at every slot of every galaxy, which stored exactly would take hundreds of gigabytes; the site stores a compact form, described in :doc:`reference`. ``compression_quality`` checks that searching the compact form ranks galaxies as the exact predictions would. For each mode it predicts a sample of galaxies exactly, runs queries of several sizes on both forms, and reports the stored size per galaxy and, over queries, the median and 5th percentile of the Spearman rank correlation between every galaxy's two scores, the median fraction of the exact top 10 kept, and ``search_quality``'s identity and redshift measures under both forms. No report from the current script is committed yet.

Request speed
-------------

``backend_performance`` starts a temporary copy of the server and times requests to it from a separate container, including Modal's network but not the visitor's. For ``/meta``, a similarity search, a text search, a galaxy's image, its spectrum and its catalogue table, it times the first request of each kind to a fresh server (*cold*, paying for what that request loads first) and then 30 more to the running server (*warm*). No report from the current script is committed yet.
