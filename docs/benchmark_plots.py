import json
from pathlib import Path

from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    Div,
    FactorRange,
    TableColumn,
    Whisker,
)
from bokeh.palettes import Category10_10, Category20_20
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
    return DataTable(
        source=source,
        columns=columns,
        index_position=None,
        height=height or 30 + 26 * len(rows),
        **WIDTH,
    )


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
    source = ColumnDataSource(
        {key: [row[key] for row in rows] for key in ("neighbours", "metric", "value")}
    )
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
        tooltips=[
            ("average precision", "@average_precision"),
            ("base rate", "@base_rate"),
        ],
        **WIDTH,
    )
    chart.hbar(y="factor", right="average_precision", height=0.8, source=source)
    chart.scatter(
        x="base_rate",
        y="factor",
        marker="dash",
        angle=1.5708,
        size=18,
        color="black",
        source=source,
    )
    names = [name for name in rows[0] if name != "centring"]
    names.insert(2, "centring")
    return column(
        heading(found, f"{found['galaxies']:,} held-out galaxies", cpus(found)),
        chart,
        table(rows, names, height=400),
        **WIDTH,
    )


def search_quality():
    found = report("search_quality")
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
        tooltips=[
            ("mean", "@mean"),
            ("interval", "@low to @high"),
            ("queries", "@queries"),
        ],
        **WIDTH,
    )
    chart.scatter(x="mean", y="factor", size=8, source=source)
    chart.add_layout(
        Whisker(
            source=source, base="factor", lower="low", upper="high", dimension="width"
        )
    )
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
    rows = [
        {
            "mode": mode,
            "selected slots": size,
            "bytes per galaxy": round(measured["bytes_per_galaxy"]),
            "spearman median": round(figures["spearman_median"], 4),
            "spearman 5th percentile": round(figures["spearman_p5"], 4),
            "top 10 median": round(figures["top10_median"], 4),
            "identity change": round(
                figures["identity_exact"] - figures["identity"], 4
            ),
            "relevance change": None
            if figures["relevance"] is None
            else round(abs(figures["relevance"] - figures["relevance_exact"]), 4),
        }
        for mode, measured in found["modes"].items()
        for size, figures in measured["sizes"].items()
    ]
    source = ColumnDataSource({name: [row[name] for row in rows] for name in rows[0]})
    source.data["factor"] = list(
        zip(source.data["mode"], source.data["selected slots"])
    )
    chart = figure(
        x_range=FactorRange(*source.data["factor"]),
        y_range=(0, 1.05),
        height=320,
        title="Stored predictions: top-10 overlap with exact rankings, by mode and selection size",
        tooltips=[
            ("Spearman median", "@{spearman median}"),
            ("top 10 median", "@{top 10 median}"),
        ],
        **WIDTH,
    )
    chart.vbar(x="factor", top="top 10 median", width=0.8, source=source)
    chart.xaxis.major_label_orientation = 1.2
    first = next(iter(found["modes"].values()))
    return column(
        heading(
            found,
            f"{first['queries']} queries over {first['sample']:,} sampled galaxies per mode",
        ),
        chart,
        table(rows, list(rows[0])),
        **WIDTH,
    )


def token_embedding_quality():
    found = report("token_embedding_quality")
    if found is None:
        return missing()
    ranks = found["ranks"]
    curves = {name: matrix["stored"] for name, matrix in found["matrices"].items()} | {
        f"{name}, combined": combined for name, combined in found["combined"].items()
    }
    chart = figure(
        x_axis_type="log",
        x_axis_label="k",
        y_range=(0, 1.05),
        height=420,
        title="Share of the exact top 10 found with the stored vectors, by k",
        **WIDTH,
    )
    for (name, curve), colour in zip(curves.items(), Category20_20):
        chart.line(
            ranks,
            [curve[str(rank)]["top10"] for rank in ranks],
            legend_label=name,
            color=colour,
            line_width=2,
        )
    chart.legend.location = "bottom_right"
    chart.legend.click_policy = "hide"
    last = str(ranks[-1])
    rows = [
        {
            "matrix": name,
            f"top 10 at k = {last}": matrix["stored"][last]["top10"],
            f"top 100 at k = {last}": matrix["stored"][last]["top100"],
            f"positive MDS top 10 at k = {last}": matrix["positive"][last]["top10"],
            f"energy at k = {last}": matrix["energy"][last],
            "negative share": matrix["negative_share"],
            "antisymmetric share": matrix.get("antisymmetric_share"),
            "symmetric part's top 10": matrix.get("symmetric", {}).get("top10"),
        }
        for name, matrix in found["matrices"].items()
    ]
    return column(
        heading(
            found,
            f"every one of {found['galaxies']:,} galaxies as a query",
            found["device"],
        ),
        chart,
        table(rows, list(rows[0])),
        **WIDTH,
    )


def backend_performance():
    found = report("backend_performance")
    if found is None:
        return missing()
    requests = found["requests"]
    rows = [
        {"request": name, "cold_ms": round(requests["cold_ms"][name], 1)}
        | {key: round(value, 1) for key, value in figures.items()}
        for name, figures in requests["warm"].items()
    ]
    source = ColumnDataSource(
        {
            "factor": [
                (row["request"], kind) for row in rows for kind in ("cold", "warm p50")
            ],
            "ms": [value for row in rows for value in (row["cold_ms"], row["p50_ms"])],
        }
    )
    chart = figure(
        y_range=FactorRange(*reversed(source.data["factor"])),
        x_axis_type="log",
        x_axis_label="latency (ms, log scale)",
        height=60 + 24 * len(source.data["factor"]),
        title="First request to a fresh server (cold) and to a running one (warm median)",
        tooltips=[("ms", "@ms")],
        **WIDTH,
    )
    chart.hbar(y="factor", left=1, right="ms", height=0.8, source=source)
    server = found["server"]
    return column(
        heading(
            found,
            f"server {server['cpu']} CPU, {server['memory_mb'][0]:,} MB requested",
        ),
        chart,
        table(rows, ("request", "cold_ms", "p50_ms", "p95_ms", "runs")),
        **WIDTH,
    )


def search_performance():
    found = report("search_performance")
    if found is None:
        return missing()
    stages_by_round = [
        (name, str(position), measured["stages"])
        for name, rounds in found["stages"]["rounds"].items()
        for position, measured in enumerate(rounds)
    ]
    stages = list(stages_by_round[0][2])
    source = ColumnDataSource(
        {"factor": [(name, position) for name, position, _ in stages_by_round]}
        | {
            stage: [measured[stage]["p50_ms"] for _, _, measured in stages_by_round]
            for stage in stages
        }
    )
    chart = figure(
        x_range=FactorRange(*source.data["factor"]),
        height=320,
        y_axis_label="median time (ms)",
        title="Search stages by version and round",
        tooltips="$name: @$name ms",
        **WIDTH,
    )
    chart.vbar_stack(
        stages,
        x="factor",
        width=0.8,
        color=Category10_10[: len(stages)],
        legend_label=stages,
        source=source,
    )
    current = found["stages"]["rounds"]["current"][0]
    timings = (
        {
            "agreement matrix build": current["matrices"]["agreements"],
            "Population map": current["matrices"]["saliency"],
        }
        | {
            f"whole search, {matches.replace('=', ' ')}": figures
            for matches, figures in current["searches"].items()
        }
        | {
            f"any-position {name.replace('_', ' ')}": figures
            for name, figures in current["anywhere"].items()
        }
    )
    versions = ", ".join(
        f"{name} <code>{commit[:7]}</code>" for name, commit in found["commits"].items()
    )
    return column(
        heading(found, f"versions: {versions}", cpus(found["stages"])),
        chart,
        Div(text="Current commit, from a running server:", **WIDTH),
        table(
            [{"timing": name} | figures for name, figures in timings.items()],
            ("timing", "p50_ms", "p95_ms", "runs"),
        ),
        **WIDTH,
    )
