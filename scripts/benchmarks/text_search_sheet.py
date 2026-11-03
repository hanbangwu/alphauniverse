import base64
import html
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

from app.config import DATASET_REVISION, artifact
from app.text_search import embed_queries, photos, query_directions, text_vectors
from app.text_views import training
from modal_app import CACHE_PATH, build_image, cache_volume
from scripts.benchmarks.common import git

app = modal.App("alphauniverse-text-search-sheet")
image = build_image.add_local_python_source("modal_app")

QUERIES = (
    "a galaxy",
    "a spiral galaxy",
    "a red spiral galaxy",
    "a red barred spiral galaxy with a ring",
    "an elliptical galaxy",
    "a round red elliptical galaxy",
    "a giant round red elliptical galaxy with a bright core",
    "an edge-on disk galaxy",
    "a merging galaxy with tidal tails",
    "an irregular galaxy",
    "a galaxy spectrum",
    "a galaxy spectrum with emission lines",
    "a galaxy spectrum with strong H-alpha emission",
    "a spectrum with strong H-alpha and [OIII] emission lines at redshift 0.1",
    "an astronomical object",
    "a star-forming galaxy",
    "a quiescent galaxy with no ongoing star formation",
    "a massive galaxy",
    "a dwarf galaxy",
    "a nearby galaxy at low redshift",
    "a distant galaxy at high redshift",
)
PAGE = 8
THUMBNAIL = 112


def encoded(values: np.ndarray) -> str:
    return base64.b64encode(values.tobytes()).decode()


def pages(scores: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        query: {
            "top": np.argsort(-score)[:PAGE].tolist(),
            "shade": encoded((rankdata(score) / len(score) * 255).astype(np.uint8)),
        }
        for query, score in scores.items()
    }


def production_pages() -> dict[str, dict[str, Any]]:
    stored = text_vectors()
    directions = query_directions(embed_queries(list(QUERIES)), stored.query_mean)
    return pages(dict(zip(QUERIES, (directions @ stored.vectors.T).numpy())))


def thumbnail(picture) -> str:
    picture = picture.convert("RGB")
    picture.thumbnail((THUMBNAIL, THUMBNAIL))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def described(galaxies: list[int]) -> dict[str, Any]:
    chosen = np.asarray(sorted(set(galaxies)))
    pictures = [picture for batch in photos(chosen) for picture in batch]
    magnitude = pq.read_table(artifact("properties"), columns=["r_magnitude_measured"])
    magnitude = magnitude.column(0).to_numpy(zero_copy_only=False)[chosen]
    points = pq.read_table(artifact("mean_points"), columns=["x", "y"])
    coordinates = np.stack([points.column(name).to_numpy() for name in ("x", "y")], 1)
    low, high = np.percentile(coordinates, [0.5, 99.5], axis=0)
    coordinates = np.clip(coordinates, low, high)
    trained = np.isin(chosen, training())
    return {
        "map": encoded(((coordinates - low) / (high - low) * 65535).astype(np.uint16)),
        "galaxies": {
            str(galaxy): {
                "image": thumbnail(picture),
                "r": None if np.isnan(r) else round(float(r), 1),
                "trained": bool(flag),
            }
            for galaxy, picture, r, flag in zip(chosen, pictures, magnitude, trained)
        },
    }


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=(16 * 1024, 48 * 1024),
    timeout=30 * 60,
    volumes={CACHE_PATH: cache_volume},
)
def sheet(earlier: list[int]) -> dict[str, Any]:
    found = production_pages()
    shown = [galaxy for page in found.values() for galaxy in page["top"]]
    return {"pages": found, **described(shown + earlier)}


def page(versions: dict[str, dict[str, Any]], shared: dict[str, Any]) -> str:
    data = json.dumps({"versions": versions, **shared})
    names = "".join(f"<li>{html.escape(name)}</li>" for name in versions)
    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text Search Sheet</title>
<style>
:root {{ --bg: #f6f7f9; --panel: #fff; --fg: #1b2230; --muted: #5d6676; --rule: #d9dee6; --mark: #c2410c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #0f131a; --panel: #171c25; --fg: #e4e8ef; --muted: #98a1b0; --rule: #2a3240; --mark: #fb923c; }} }}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.5 system-ui, sans-serif; margin: 0; }}
main {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 14px; }}
.row {{ display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 10px; align-items: start; border-top: 1px solid var(--rule); padding-top: 10px; }}
.row canvas {{ width: 150px; height: 150px; background: var(--panel); border-radius: 4px; }}
.strip {{ display: grid; grid-auto-flow: column; grid-auto-columns: 112px; gap: 6px; overflow-x: auto; }}
figure {{ margin: 0; font: 11px ui-monospace, monospace; color: var(--muted); }}
figure img {{ width: 112px; height: 112px; border-radius: 4px; display: block; }}
figure.trained img {{ outline: 3px solid var(--mark); outline-offset: -3px; }}
.label {{ font-weight: 600; grid-column: 1 / -1; }}
select {{ font: inherit; max-width: 100%; }}
</style>
<main>
<h1>Text search sheet</h1>
<p>Each row is one version's 8 best matches out of every galaxy, with its r magnitude; an outlined tile trained the text model. The map is the site's AION map, each galaxy shaded by its score percentile, darkest highest, with the 8 best matches in orange.</p>
<ul>{names}</ul>
<label>Query <select id="query"></select></label>
<div id="rows"></div>
</main>
<script>
const data = {data};
const bytes = text => Uint8Array.from(atob(text), c => c.charCodeAt(0));
const map = new Uint16Array(bytes(data.map).buffer);
const select = document.getElementById("query");
const holder = document.getElementById("rows");
const queries = [...new Set(Object.values(data.versions).flatMap(found => Object.keys(found)))];
for (const query of queries) select.add(new Option(query, query));
function draw(canvas, shade, top) {{
  const size = 300, context = canvas.getContext("2d");
  canvas.width = canvas.height = size;
  const order = [...shade.keys()].sort((a, b) => shade[a] - shade[b]);
  for (const galaxy of order) {{
    const level = shade[galaxy] / 255;
    context.fillStyle = `rgb(${{Math.round(220 - 200 * level)}}, ${{Math.round(228 - 150 * level)}}, ${{Math.round(240 - 70 * level)}})`;
    context.fillRect(map[2 * galaxy] / 65535 * (size - 3), (1 - map[2 * galaxy + 1] / 65535) * (size - 3), 3, 3);
  }}
  context.fillStyle = "#c2410c";
  for (const galaxy of top) {{
    context.beginPath();
    context.arc(map[2 * galaxy] / 65535 * (size - 3) + 1.5, (1 - map[2 * galaxy + 1] / 65535) * (size - 3) + 1.5, 5, 0, 2 * Math.PI);
    context.fill();
  }}
}}
function show() {{
  holder.replaceChildren();
  for (const [name, found] of Object.entries(data.versions)) {{
    const entry = found[select.value];
    if (!entry) continue;
    const row = document.createElement("section");
    row.className = "row";
    const tiles = entry.top.map(galaxy => {{
      const about = data.galaxies[galaxy];
      return `<figure class="${{about.trained ? "trained" : ""}}"><img alt="galaxy ${{galaxy}}" src="data:image/jpeg;base64,${{about.image}}"><figcaption>#${{galaxy}} r ${{about.r ?? "?"}}</figcaption></figure>`;
    }}).join("");
    row.innerHTML = `<div class="label"></div><canvas></canvas><div class="strip">${{tiles}}</div>`;
    row.querySelector(".label").textContent = name;
    draw(row.querySelector("canvas"), bytes(entry.shade), entry.top);
    holder.append(row);
  }}
}}
select.addEventListener("change", show);
show();
</script>
"""


@app.local_entrypoint()
def main(out: str = "text_search_sheet.html", earlier: str = "") -> None:
    versions = json.loads(Path(earlier).read_text())["versions"] if earlier else {}
    shown = [
        galaxy
        for found in versions.values()
        for entry in found.values()
        for galaxy in entry["top"]
    ]
    found = sheet.remote(shown)
    name = f"{git('describe', '--always', '--dirty')}, {DATASET_REVISION[:7]}, {datetime.now(UTC):%Y-%m-%d}"
    versions[name] = found.pop("pages")
    Path(out).with_suffix(".json").write_text(json.dumps({"versions": versions}))
    Path(out).write_text(page(versions, found))
    print(f"wrote {out}")
