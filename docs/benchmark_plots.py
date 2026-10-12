import json
from pathlib import Path

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, DataTable, Div, FactorRange, TableColumn, Whisker
from bokeh.palettes import Category10_10
from bokeh.plotting import figure

REPORTS = Path(__file__).parent / "benchmarks"
WIDTH = {"sizing_mode": "stretch_width"}


def report(script):
    path = REPORTS / f"{script}.json"
    return json.loads(path.read_text()) if path.exists() else None


def heading(found, *details):
    parts = [f"Commit <code>{found['commit']}</code>, {found['date'][:10]}", *details]
    return Div(text="; ".join(parts) + ".", **WIDTH)


def table(rows, names, height=None):
    source = ColumnDataSource({name: [row[name] for row in rows] for name in names})
    columns = [TableColumn(field=name, title=name) for name in names]
    return DataTable(source=source, columns=columns, index_position=None, height=height or 30 + 26 * len(rows), **WIDTH)


def missing():
    return Div(text="No report is committed yet.", **WIDTH)


def cpus(found):
    return f"{found['environment']['cpu_count']} CPUs visible to the container"


def projection_quality():
    found = report("projection_quality")
    if found is None:
        return missing()
    rows = [
        {"neighbours": neighbours, "metric": metric, "value": value}
        for neighbours, values in found["neighbours"].items()
        for metric, value in values.items()
    ]
    source = ColumnDataSource({key: [row[key] for row in rows] for key in ("neighbours", "metric", "value")})
    source.data["factor"] = list(zip(source.data["neighbours"], source.data["metric"]))
    chart = figure(
        x_range=FactorRange(*source.data["factor"]),
        y_range=(0, 1),
        height=300,
        title="Neighbours kept, by k",
        tooltips=[("k", "@neighbours"), ("metric", "@metric"), ("value", "@value")],
        **WIDTH,
    )
    chart.vbar(x="factor", top="value", width=0.8, source=source)
    return column(
        heading(found, f"{found['size']:,} embeddings", cpus(found)),
        chart,
        table(rows, ("neighbours", "metric", "value")),
        **WIDTH,
    )


def text_search_quality():
    found = report("text_search_quality")
    if found is None:
        return missing()
    rows = [
        {"query": query, "space": space, "centring": centring} | figures
        for query, spaces in found["queries"].items()
        for space, centrings in spaces.items()
        for centring, figures in centrings.items()
    ]
    raw = [row for row in rows if row["centring"] == "raw"]
    source = ColumnDataSource(
        {
            "factor": [(row["query"], row["space"]) for row in raw],
            "average_precision": [row["average_precision"] for row in raw],
            "base_rate": [row["base_rate"] for row in raw],
        }
    )
    chart = figure(
        y_range=FactorRange(*reversed(source.data["factor"])),
        x_range=(0, 1),
        height=40 + 24 * len(raw),
        title="Average precision, raw (bars) against the base rate (ticks)",
        tooltips=[("average precision", "@average_precision"), ("base rate", "@base_rate")],
        **WIDTH,
    )
    chart.hbar(y="factor", right="average_precision", height=0.8, source=source)
    chart.scatter(x="base_rate", y="factor", marker="dash", angle=1.5708, size=18, color="black", source=source)
    names = [name for name in rows[0] if name != "centring"]
    names.insert(2, "centring")
    return column(
        heading(found, f"{found['galaxies']:,} held-out galaxies", cpus(found)),
        chart,
        table(rows, names, height=400),
        **WIDTH,
    )


def pql_quality():
    found = report("pql_quality")
    if found is None:
        return missing()
    rows = [
        {"metric": metric, "selection": selection} | figures
        for selection, metrics in found["results"].items()
        for metric, figures in metrics.items()
    ]
    rows.sort(key=lambda row: row["metric"])
    source = ColumnDataSource({name: [row[name] for row in rows] for name in rows[0]})
    source.data["factor"] = list(zip(source.data["metric"], source.data["selection"]))
    chart = figure(
        y_range=FactorRange(*reversed(source.data["factor"])),
        height=40 + 26 * len(rows),
        title="Mean over queries (dot) with its interval (whisker)",
        tooltips=[("mean", "@mean"), ("interval", "@low to @high"), ("queries", "@queries")],
        **WIDTH,
    )
    chart.scatter(x="mean", y="factor", size=8, source=source)
    chart.add_layout(Whisker(source=source, base="factor", lower="low", upper="high", dimension="width"))
    return column(
        heading(found, f"{found['sample']:,} sampled galaxies", cpus(found)),
        chart,
        table(rows, ("metric", "selection", "mean", "low", "high", "queries")),
        **WIDTH,
    )


def compression_quality():
    found = report("compression_quality")
    if found is None:
        return missing()
    rows = []
    for mode, measured in found["modes"].items():
        current = measured["schemes"]["current"]
        for size, sides in current["sizes"].items():
            figures = sides["gallery"]
            rows.append(
                {
                    "mode": mode,
                    "selected slots": size,
                    "bytes per galaxy": current["bytes_per_galaxy"],
                    "spearman median": round(figures["spearman_median"], 4),
                    "top 10 median": round(figures["top10_median"], 4),
                    "identity change": round(figures["identity_exact"] - figures["identity"], 4),
                    "relevance change": round(abs(figures["relevance"] - figures["relevance_exact"]), 4),
                }
            )
    source = ColumnDataSource({name: [row[name] for row in rows] for name in rows[0]})
    source.data["factor"] = list(zip(source.data["mode"], source.data["selected slots"]))
    chart = figure(
        x_range=FactorRange(*source.data["factor"]),
        y_range=(0, 1.05),
        height=320,
        title="Stored scheme: top-10 overlap with exact rankings, by mode and selection size",
        tooltips=[("Spearman median", "@{spearman median}"), ("top 10 median", "@{top 10 median}")],
        **WIDTH,
    )
    chart.vbar(x="factor", top="top 10 median", width=0.8, source=source)
    chart.xaxis.major_label_orientation = 1.2
    targets = ", ".join(f"{name} {value}" for name, value in found["targets"].items())
    choices = [{"mode": mode, "most compact scheme meeting the targets": scheme} for mode, scheme in found["choices"]["accuracy"].items()]
    return column(
        heading(found, f"targets: {targets}"),
        chart,
        table(rows, list(rows[0])),
        table(choices, list(choices[0])),
        **WIDTH,
    )


def backend_performance():
    found = report("backend_performance")
    if found is None:
        return missing()
    requests = found["requests"]
    warm = requests["warm"]
    source = ColumnDataSource(
        {
            "route": list(warm),
            "p50_ms": [figures["p50_ms"] for figures in warm.values()],
            "p95_ms": [figures["p95_ms"] for figures in warm.values()],
            "runs": [figures["runs"] for figures in warm.values()],
        }
    )
    chart = figure(
        y_range=list(reversed(warm)),
        x_range=(0, 1.05 * max(source.data["p95_ms"])),
        x_axis_label="latency (ms)",
        height=60 + 24 * len(warm),
        title="Warm requests: median (bar) to 95th percentile (whisker)",
        tooltips=[("p50", "@p50_ms ms"), ("p95", "@p95_ms ms"), ("runs", "@runs")],
        **WIDTH,
    )
    chart.hbar(y="route", right="p50_ms", height=0.6, source=source)
    chart.add_layout(Whisker(source=source, base="route", lower="p50_ms", upper="p95_ms", dimension="width"))
    cold = [{"request": name.removeprefix("cold_").removesuffix("_ms"), "ms": value} for name, value in requests.items() if name.startswith("cold_")]
    server = found["server"]
    return column(
        heading(found, f"server {server['cpu']} CPU, {server['memory_mb'][0]:,} MB requested"),
        chart,
        table([{"route": route} | figures for route, figures in warm.items()], ("route", "p50_ms", "p95_ms", "runs")),
        Div(text="First request to a freshly started server:", **WIDTH),
        table(cold, ("request", "ms")),
        **WIDTH,
    )


def search_performance():
    found = report("search_performance")
    if found is None:
        return missing()
    rounds = found["stages"]["rounds"]
    stages_by_round = [
        (name, str(position), measured["stages"])
        for name, entries in rounds.items()
        for position, measured in enumerate(entries)
        if "stages" in measured
    ]
    failed = [name for name, entries in rounds.items() if not any("stages" in measured for measured in entries)]
    stages = list(stages_by_round[0][2])
    source = ColumnDataSource(
        {"factor": [(name, position) for name, position, _ in stages_by_round]}
        | {stage: [measured[stage]["p50_ms"] for _, _, measured in stages_by_round] for stage in stages}
    )
    chart = figure(
        x_range=FactorRange(*source.data["factor"]),
        height=320,
        y_axis_label="median time (ms)",
        title="Search stages by version and round",
        tooltips="$name: @$name ms",
        **WIDTH,
    )
    chart.vbar_stack(stages, x="factor", width=0.8, color=Category10_10[: len(stages)], legend_label=stages, source=source)
    versions = ", ".join(
        f"{name} <code>{commit[:7]}</code>{' (failed)' if name in failed else ''}" for name, commit in found["commits"].items()
    )
    return column(heading(found, f"versions: {versions}", cpus(found["stages"])), chart, **WIDTH)
