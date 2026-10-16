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


def template_embedding_quality():
    found = report("template_embedding_quality")
    if found is None:
        return missing()
    ranks = found["ranks"]
    exact = {"score": "S", "share": "S / S_sat", "fit": "S − S_sat"}
    truths = exact | {"template": "the template Gram"}
    charts = []
    for truth, label in truths.items():
        chart = figure(
            x_axis_type="log",
            x_axis_label="k",
            y_range=(0, 1.05),
            height=360,
            title=f"Share of the top 10 by {label} found with the stored vectors, by k",
            **WIDTH,
        )
        for (name, block), colour in zip(found["blocks"].items(), Category20_20):
            if truth in block["stored"][str(ranks[0])]:
                chart.line(
                    ranks,
                    [block["stored"][str(rank)][truth]["top10"] for rank in ranks],
                    legend_label=name,
                    color=colour,
                    line_width=2,
                )
        chart.legend.location = "bottom_right"
        chart.legend.click_policy = "hide"
        charts.append(chart)
    last = str(ranks[-1])
    rows = [
        {
            "block": name,
            "galleries": block["galleries"],
            **{
                f"own rank by {label}, median": block["own_rank"]
                .get(truth, {})
                .get("p50")
                for truth, label in exact.items()
            },
            **{
                f"own data in top 10 by {label}": block["own_rank"]
                .get(truth, {})
                .get("top10")
                for truth, label in exact.items()
            },
            **{
                f"template Gram's top 10 by {label}": block["template_gram"]
                .get(truth, {})
                .get("top10")
                for truth, label in exact.items()
            },
            f"energy at k = {last}": block["energy"][last],
        }
        for name, block in found["blocks"].items()
    ]
    return column(
        heading(
            found,
            f"every one of {found['galaxies']:,} galaxies as a query",
            found["device"],
        ),
        *charts,
        table(rows, list(rows[0])),
        **WIDTH,
    )


def recalled(exact, reverse):
    return {
        "R@1": exact["r1"],
        "R@10": exact["r10"],
        "median rank": exact["median"],
        "reverse R@1": reverse["r1"],
        "reverse R@10": reverse["r10"],
        "reverse median rank": reverse["median"],
    }


def retrieval_quality():
    found = report("retrieval_quality")
    if found is None:
        return missing()
    ranks = [str(rank) for rank in found["ranks"]]
    rows, full = [], []
    for test, measured in found["tests"].items():
        row = {
            "set": test,
            "galaxies": measured["galaxies"],
            "score": "AION",
        } | recalled(**measured["aion"])
        rows.append(row | {f"R@10 at k = {rank}": "" for rank in ranks})
        full.append(row)
        for role in ("token", "template"):
            for group in ("combined", "blocks"):
                for name, exact in measured[role]["exact"][group].items():
                    row = {
                        "set": test,
                        "galaxies": measured["galaxies"],
                        "score": f"{role} {name}",
                    } | recalled(exact, measured[role]["reverse"][group][name])
                    rows.append(
                        row
                        | {
                            f"R@10 at k = {rank}": measured[role][rank][group][name][
                                "r10"
                            ]
                            for rank in ranks
                        }
                    )
                    if group == "combined":
                        full.append(row)
    source = ColumnDataSource(
        {
            "factor": [(row["set"], row["score"]) for row in full],
            "r1": [row["R@1"] for row in full],
            "r10": [row["R@10"] for row in full],
            "reverse": [row["reverse R@10"] for row in full],
        }
    )
    chart = figure(
        x_range=FactorRange(*source.data["factor"]),
        y_range=(0, 1.05),
        height=420,
        title="Share in the top 10 by full-object scores: each galaxy, ranked from its encoding (bars), and each encoding, ranked from its galaxy (dots)",
        tooltips=[("R@1", "@r1"), ("R@10", "@r10"), ("reverse R@10", "@reverse")],
        **WIDTH,
    )
    chart.vbar(x="factor", top="r10", width=0.8, source=source)
    chart.scatter(x="factor", y="reverse", size=8, color="black", source=source)
    chart.xaxis.major_label_orientation = 0.8
    return column(
        heading(found, f"ranked among {found['galaxies']:,} galaxies", found["device"]),
        chart,
        table(rows, list(rows[0]), height=600),
        **WIDTH,
    )


def drawn(probe, shots):
    found = {}
    for count, shot in shots.items():
        found[f"{probe} r2, {count} fitted"] = f"{shot['r2']} ± {shot['sd']}"
        if "alpha" in shot:
            found[f"{probe} α, {count} fitted"] = (
                f"{shot['alpha']:.3g}, {shot['alpha_at_edge']:.0%} at an end"
            )
    return found


def redshift_quality():
    found = report("redshift_quality")
    if found is None:
        return missing()
    rows = [
        {"input": name, "survey": survey} | figures
        for name, surveys in found["inputs"].items()
        for survey, figures in surveys.items()
    ]
    return column(
        heading(found, found["device"]),
        table(rows, ("input", "survey", "galaxies", "bias", "nmad", "outlier_rate")),
        **WIDTH,
    )


def probe_quality():
    found = report("probe_quality")
    if found is None:
        return missing()
    rows, curves, modes = [], [], []
    for target, measured in found["probes"].items():
        features = {"AION": measured["aion"]} | {
            f"{role} {kind}": probes
            for role in ("token", "template")
            for kind, probes in measured[role].items()
        }
        curve = figure(
            x_axis_type="log",
            x_axis_label="galaxies fitted",
            y_axis_label="R²",
            height=300,
            title=f"{target}: ridge R² on a held-out fifth, mean and standard deviation of {found['draws']} draws",
            **WIDTH,
        )
        for (name, probes), colour in zip(features.items(), Category20_20):
            row = {"target": target, "features": name, "galaxies": measured["galaxies"]}
            for probe, metrics in probes.items():
                row |= {
                    f"{probe} r2": metrics["r2"],
                    f"{probe} rms": metrics["rms"],
                } | drawn(probe, metrics["shots"])
            rows.append(row)
            shots = probes["ridge"]["shots"]
            spread = ColumnDataSource(
                {
                    "count": [int(count) for count in shots],
                    "r2": [shot["r2"] for shot in shots.values()],
                    "lower": [shot["r2"] - shot["sd"] for shot in shots.values()],
                    "upper": [shot["r2"] + shot["sd"] for shot in shots.values()],
                }
            )
            curve.line(
                "count",
                "r2",
                source=spread,
                legend_label=name,
                color=colour,
                line_width=2,
            )
            curve.add_layout(
                Whisker(
                    base="count",
                    lower="lower",
                    upper="upper",
                    source=spread,
                    line_color=colour,
                )
            )
        curve.legend.location = "bottom_right"
        curves.append(curve)
        modes.extend(
            {
                "target": target,
                "features": role,
                "mode": name,
                "stacked weight, all galaxies": entry["weight"],
            }
            | drawn("ridge", entry["shots"])
            for role, fits in measured["modes"].items()
            for name, entry in fits.items()
        )
    source = ColumnDataSource(
        {
            "factor": [(row["target"], row["features"]) for row in rows],
            "ridge": [row["ridge r2"] for row in rows],
            "knn": [row.get("knn r2") for row in rows],
        }
    )
    chart = figure(
        x_range=FactorRange(*source.data["factor"]),
        height=420,
        title=f"R² of ridge regression (bars) and of the {found['neighbours']} nearest neighbours (dots)",
        tooltips=[("ridge", "@ridge"), ("neighbours", "@knn")],
        **WIDTH,
    )
    chart.vbar(x="factor", top="ridge", width=0.8, source=source)
    chart.scatter(x="factor", y="knn", size=8, color="black", source=source)
    chart.xaxis.major_label_orientation = 0.8
    return column(
        heading(
            found,
            f"{found['rank']} components per mode",
            f"pca_n: the first n principal components of all modes' components, n = {', '.join(map(str, found['widths']))}",
            "stacked: ridge on each mode, weighted by a non-negative fit to their cross-validated predictions",
            f"{found['folds']}-fold cross-validation",
            cpus(found),
        ),
        chart,
        *curves,
        table(
            [{name: row.get(name, "") for name in rows[0]} for row in rows],
            list(rows[0]),
        ),
        table(modes, list(modes[0]), height=600),
        **WIDTH,
    )


def neighbour_quality():
    found = report("neighbour_quality")
    if found is None:
        return missing()
    rankings = [
        {"ranking": name}
        | {
            f"{target} |Δ|, median": values["median"]
            for target, values in entry.items()
        }
        | {
            f"{target}, beats random": values["beats_random"]
            for target, values in entry.items()
        }
        for name, entry in found["physics"]["rankings"].items()
    ]
    geometry = found["geometry"]
    agreement = [
        {
            "features": f"{role} {kind}",
            "search": name,
            f"share of its top {found['neighbours']}": value,
        }
        for role in ("token", "template")
        for kind in ("blocks", "joint")
        for name, value in geometry[role][kind].items()
    ]
    counts = [str(count) for count in found["preserved"]]
    shares = [
        {"factor": (name, search)} | kept
        for name, searches in found["maps"].items()
        for search, kept in searches.items()
    ]
    source = ColumnDataSource({key: [row[key] for row in shares] for key in shares[0]})
    kept = [
        {"map": map_name, "search": search}
        | {f"{count} nearest kept": row[count] for count in counts}
        for row in shares
        for map_name, search in [row["factor"]]
    ]
    chart = figure(
        x_range=FactorRange(*source.data["factor"]),
        y_range=(0, 1.05),
        height=380,
        title=f"Share of each galaxy's {counts[0]} (bars) and {counts[1]} (dots) nearest by each search that a map keeps nearest",
        tooltips=[(f"{count} nearest", f"@{{{count}}}") for count in counts],
        **WIDTH,
    )
    chart.vbar(x="factor", top=counts[0], width=0.8, source=source)
    chart.scatter(x="factor", y=counts[1], size=8, color="black", source=source)
    chart.xaxis.major_label_orientation = 0.8
    return column(
        heading(
            found,
            f"{found['rank']} components per mode",
            f"joint: all modes' components side by side; token_n: the token one's first n principal components, n = {', '.join(map(str, found['widths']))}",
            f"{found['physics']['galaxies']:,} PROVABGS galaxies for the physics",
            f"{geometry['queries']:,} queries for the geometry, where chance shares {geometry['random']}",
            cpus(found),
        ),
        chart,
        table(kept, list(kept[0])),
        table(rankings, list(rankings[0])),
        table(agreement, list(agreement[0]), height=600),
        **WIDTH,
    )


def fitted(measured):
    return {
        "galaxies": measured["galaxies"],
        **{
            f"{model} χ²ν, median": values["p50"]
            for model, values in measured["chi2"].items()
        },
        "recovered, median": measured["recovered"]["p50"],
    }


def template_quality():
    found = report("template_quality")
    if found is None:
        return missing()
    modes = found["modes"]
    source = ColumnDataSource(
        {
            "mode": list(modes),
            **{
                share: [measured["recovered"][share] for measured in modes.values()]
                for share in ("p5", "p50", "p95")
            },
        }
    )
    recovered = figure(
        x_range=list(modes),
        height=300,
        title="Share recovered, median and 5th to 95th percentile over galaxies",
        **WIDTH,
    )
    recovered.vbar(x="mode", top="p50", width=0.8, source=source)
    recovered.segment(
        x0="mode", y0="p5", x1="mode", y1="p95", color="black", source=source
    )
    recovered.xaxis.major_label_orientation = 0.8
    charts = [recovered]
    for key in ("radius", "rest_wavelength"):
        chart = figure(
            height=300,
            x_axis_label=key.replace("_", " "),
            y_axis_label="mean normalised residual",
            title=f"Residuals by {key.replace('_', ' ')}",
            **WIDTH,
        )
        lines = [
            (mode, model)
            for mode, measured in modes.items()
            if key in measured
            for model in ("template", "floor")
        ]
        for (mode, model), colour in zip(lines, Category10_10):
            chart.line(
                modes[mode][key]["centres"],
                modes[mode][key][model],
                legend_label=f"{mode} {model}",
                color=colour,
                line_width=2,
            )
        chart.legend.click_policy = "hide"
        charts.append(chart)
    rows = [{"mode": mode} | fitted(measured) for mode, measured in modes.items()]
    pulls = [
        {
            "value": name,
            **{f"{model}, median |pull|": value for model, value in values.items()},
        }
        for measured in modes.values()
        for name, values in measured.get("median_absolute_pull", {}).items()
    ]
    quartiles = [
        {"mode": mode, "S/N per pixel": " to ".join(map(str, quartile["snr"]))}
        | fitted(quartile)
        for mode, measured in modes.items()
        for quartile in measured.get("by_snr", [])
    ]
    return column(
        heading(found, found["device"]),
        *charts,
        table(rows, list(rows[0])),
        table(pulls, list(pulls[0])),
        table(quartiles, list(quartiles[0])),
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
