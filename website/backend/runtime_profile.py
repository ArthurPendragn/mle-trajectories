"""Runtime profile of one ``runtime_stats_*.json`` store, for the run page.

Read straight from the store (no pipeline loading): a per-pipeline table and the
per-operator-class time table ("where did the time go"), as the static report's
``_runtime_section`` / ``_heavy_hitters_section`` computed them.
"""
from __future__ import annotations

import json
from pathlib import Path

from .registry import Run, RuntimeStore, _code_sha1


def _op_class(label: str, known: set[str]) -> str:
    """Operator class behind a measured op label. The store keys ops by stratum's
    readable label, which drops the ``Op`` suffix on a few classes
    (``Predictor``, ``Split``, ...); restore it against the physical classes the
    analysis saw, when there is one."""
    cls = label.split("(")[0].split(" [")[0].strip()
    if not known or cls in known:
        return cls
    return f"{cls}Op" if f"{cls}Op" in known else cls


def build_profile(run: Run, store: RuntimeStore, current_commit: str | None,
                  known_classes: set[str] | None = None) -> dict:
    data = json.loads(Path(store.path).read_text())
    entries = data.get("pipelines") or {}
    meta = data.get("meta") or {}
    store_commit = (meta.get("versions") or {}).get("stratum_commit")
    source = next((s for s in [*run.sources, *([run.originals] if run.originals else [])]
                   if s.name == store.source), None)
    known = known_classes or set()

    pipelines, agg = [], {}
    for name, e in entries.items():
        mem = e.get("memory") or {}
        f = source.files.get(name) if source else None
        built = e.get("stratum_commit") or store_commit
        pipelines.append({
            "name": name, "status": e.get("status"),
            "wall_s": e.get("wall_s"), "op_time_s": e.get("op_time_s"),
            "total_s": e.get("total_s"), "max_rss_mb": e.get("max_rss_mb"),
            "mean_mb": mem.get("mean_mb"), "n_op_calls": e.get("n_op_calls"),
            "best_score": e.get("best_score"),
            "error": (e.get("error") or "").strip().splitlines()[-1:] or None,
            "code_changed": f is None or e.get("code_sha1") != _code_sha1(f),
            "old_build": bool(current_commit and built != current_commit),
        })
        if e.get("status") != "ok":
            continue
        for row in e.get("ops") or ():
            a = agg.setdefault(_op_class(row["op"], known),
                               {"time_s": 0.0, "calls": 0, "pipes": {}})
            a["time_s"] += row.get("time_s") or 0.0
            a["calls"] += row.get("count") or 0
            a["pipes"][name] = a["pipes"].get(name, 0.0) + (row.get("time_s") or 0.0)

    measured = [p for p in pipelines if p["status"] == "ok"]
    total_op = sum(a["time_s"] for a in agg.values()) or 0.0
    ops = []
    for op, a in sorted(agg.items(), key=lambda kv: -kv[1]["time_s"]):
        worst, worst_t = max(a["pipes"].items(), key=lambda kv: kv[1])
        ops.append({"op": op, "time_s": a["time_s"],
                    "share": a["time_s"] / total_op if total_op else None,
                    "calls": a["calls"],
                    "per_call_s": a["time_s"] / a["calls"] if a["calls"] else None,
                    "n_pipelines": len(a["pipes"]),
                    "heaviest": {"name": worst, "time_s": worst_t}})
    pipelines.sort(key=lambda p: -(p["wall_s"] or 0))
    return {
        "file": store.path.name, "source": store.source,
        "data": store.data, "legacy_rows": store.legacy_rows,
        "n_measured": len(measured), "n_failed": len(pipelines) - len(measured),
        "wall_total_s": sum(p["wall_s"] or 0 for p in measured),
        "op_total_s": total_op,
        "scoring": meta.get("scoring"),
        "pipelines": pipelines, "ops": ops,
    }
