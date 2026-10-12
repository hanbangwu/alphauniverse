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
   environment = report["environment"]
   show(
       column(
           Div(
               text=f"Commit <code>{report['commit']}</code>, {report['date'][:10]}; "
               f"{report['size']:,} embeddings; {environment['cpu_count']} CPUs visible to the container."
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
   environment = report["environment"]
   show(
       column(
           Div(
               text=f"Commit <code>{report['commit']}</code>, {report['date'][:10]}; "
               f"{report['galaxies']:,} held-out galaxies; {environment['cpu_count']} CPUs visible to the container."
           ),
           chart,
           table,
       )
   )

Search quality
--------------

``pql_quality`` samples galaxies with a spectrum and a redshift and measures the median :math:`|\Delta z| / (1 + z)` of each one's top 10 matches. It also hides each sampled galaxy's spectra and redshift, predicts it again, and checks whether the hidden copy ranks in the top 10, and whether merely having a spectrum, rather than resembling the query, lifts a galaxy into the top 32. Its redshift is DESI's, else SDSS's, without the flag or limit AION's input uses. No report is committed yet.

``compression_quality`` measures how each way of storing AION's predicted distributions changes the similarity scores and rankings, against the exact distributions, including the storage described in :doc:`explanation`. No report is committed yet.

Speed
-----

``search_performance`` times each stage of a similarity search in a container built like the server, comparing the current commit with ``main``. Its committed report timed the search that predictive query likelihood replaced, so no current figures are shown.

``backend_performance`` starts a temporary copy of the server and times requests to it from a separate container, including Modal's network but not the visitor's. The server runs in one container. Similarity search figures in the committed report predate the current search and are not shown.

.. bokeh-plot::

   import json
   from pathlib import Path

   from bokeh.layouts import column
   from bokeh.models import ColumnDataSource, DataTable, Div, TableColumn, Whisker
   from bokeh.plotting import figure, show

   report = json.loads(Path("benchmarks/backend_performance.json").read_text())
   routes = {
       "meta": "/meta",
       "coverage": "/galaxy/{g}",
       "image": "/galaxy/{g}/image",
       "tokens": "/galaxy/{g}/image/tokens",
       "spectrum": "/galaxy/{g}/spectrum",
       "spectrum tokens": "/galaxy/{g}/spectrum/tokens",
       "table": "/galaxy/{g}/table",
       "text search": "/search/text",
   }
   warm = report["requests"]["warm"]
   source = ColumnDataSource(
       {
           "route": list(routes.values()),
           "p50_ms": [warm[name]["p50_ms"] for name in routes],
           "p95_ms": [warm[name]["p95_ms"] for name in routes],
           "runs": [warm[name]["runs"] for name in routes],
       }
   )
   chart = figure(
       y_range=list(reversed(routes.values())),
       x_axis_label="latency (ms)",
       height=320,
       width=760,
       title="Warm requests: median (bar) to 95th percentile (whisker)",
       tooltips=[("p50", "@p50_ms ms"), ("p95", "@p95_ms ms"), ("runs", "@runs")],
   )
   chart.hbar(y="route", right="p50_ms", height=0.6, source=source)
   chart.add_layout(Whisker(source=source, base="route", lower="p50_ms", upper="p95_ms", dimension="width"))
   table = DataTable(
       source=source,
       columns=[TableColumn(field=name, title=name) for name in ("route", "p50_ms", "p95_ms", "runs")],
       index_position=None,
       width=760,
       height=260,
   )
   server = report["server"]
   show(
       column(
           Div(
               text=f"Commit <code>{report['commit']}</code>, {report['date'][:10]}; "
               f"server {server['cpu']} CPU, {server['memory_mb'][0]:,} MB requested; "
               f"first <code>/search/text</code> on a fresh server, loading EmbeddingGemma: "
               f"{report['requests']['cold_text_search_ms'] / 1000:.1f} s."
           ),
           chart,
           table,
       )
   )
