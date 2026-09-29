"""The search tree of a run, rendered by graphviz from the registry's lineage.

Needs no pipeline loading -- only the steps and their parents -- so it is drawn
for every run that has a lineage, whether or not it was ever skrubified. Nodes
whose pipeline has no plan in the selected source are drawn dashed.

Each node group carries ``data-module="<pipeline>"`` so the frontend can make it
clickable (ticking that pipeline in the operator explorer).
"""
from __future__ import annotations

import html
import re
from functools import lru_cache

from graphviz import Digraph

from .registry import Run, Source

# categorical, assigned to phases in first-seen order (as pipeline_analyzer.merged)
PHASE_COLORS = ["#2563eb", "#0d9488", "#c2410c", "#7c3aed", "#b45309",
                "#be185d", "#4d7c0f", "#0369a1", "#9333ea", "#a16207"]
NO_PHASE = "#64748b"


def _score_fill(up: float | None) -> str:
    if up is None:
        return "#e2e8f0"
    if up > 0.0005:
        return "#b7f0c6"          # improved on the parent
    if up < -0.0005:
        return "#f7c9c9"          # regressed
    return "#fde6b0"              # flat


def _tint(hex_color: str, t: float = 0.82) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * t):02x}" for c in (r, g, b))


def short_name(module: str) -> str:
    """``pipeline_05`` -> ``p05``; mlevolve's ``0002_<uuid>`` -> ``0002``."""
    if module.startswith("pipeline_"):
        return "p" + module[len("pipeline_"):]
    m = re.match(r"^(\d{3,})_[0-9a-f]{16,}$", module)
    return m.group(1) if m else module


_NODE = re.compile(r'<g id="n(\d+)" class="node">\s*<title>[^<]*</title>')


@lru_cache(maxsize=64)
def _render(dot_source: str) -> str:
    from graphviz import Source as GvSource
    raw = GvSource(dot_source).pipe(format="svg").decode()
    return raw[raw.find("<svg"):]


def build_tree(run: Run, source: Source | None) -> dict:
    steps = [s for s in run.steps if s.module]
    if not steps:
        return {"modules": [], "phase_colors": {}, "svg": {}}
    modules = [s.module for s in steps]
    index = {m: i for i, m in enumerate(modules)}
    score = {s.module: s.score for s in steps}
    lower = bool(run.metric.lower_is_better)

    phase_colors: dict[str, str] = {}
    for s in steps:
        if s.phase and s.phase not in phase_colors:
            phase_colors[s.phase] = PHASE_COLORS[len(phase_colors) % len(PHASE_COLORS)]

    def dot(color_by: str) -> str:
        g = Digraph(graph_attr={"rankdir": "TB", "bgcolor": "transparent",
                                "nodesep": "0.25", "ranksep": "0.5"},
                    node_attr={"shape": "box", "style": "filled,rounded",
                               "fontname": "Helvetica", "fontsize": "11",
                               "penwidth": "1.3", "margin": "0.12,0.06"},
                    edge_attr={"color": "#94a3b8", "arrowsize": "0.7"})
        for i, s in enumerate(steps):
            parent = score.get(s.parent) if s.parent in index else None
            delta = None if s.score is None or parent is None else s.score - parent
            up = None if delta is None else (-delta if lower else delta)
            if color_by == "phase":
                hue = phase_colors.get(s.phase or "", NO_PHASE)
                fill, border = _tint(hue), hue
            else:
                fill, border = _score_fill(up), "#334155"
            style = "filled,rounded"
            if source is not None and s.module not in source.files:
                style += ",dashed"
            sc = "—" if s.score is None else f"{s.score:.5g}"
            dl = "" if delta is None else f"\n({'+' if delta >= 0 else ''}{delta:.4g})"
            g.node(s.module, f"{short_name(s.module)}\n{sc}{dl}", id=f"n{i}",
                   fillcolor=fill, color=border, style=style)
        for s in steps:
            if s.parent in index:
                g.edge(s.parent, s.module)
        return g.source

    def annotate(svg: str) -> str:
        # graphviz cannot emit custom attributes: rewrite its node ids into a
        # data-module hook, and use the step's rationale as the hover title
        def sub(m):
            s = steps[int(m.group(1))]
            tip = f"{s.module}" + (f"\n\n{s.desc}" if s.desc else "")
            return (f'<g class="node" data-module="{html.escape(s.module, quote=True)}">'
                    f"<title>{html.escape(tip[:1500])}</title>")
        return _NODE.sub(sub, svg)

    svgs = {"delta": annotate(_render(dot("delta")))}
    if phase_colors:
        svgs["phase"] = annotate(_render(dot("phase")))
    return {"modules": modules, "phase_colors": phase_colors, "svg": svgs,
            "lower_is_better": lower}
