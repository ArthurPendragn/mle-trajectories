"""Static code analysis of a run's original scripts (``tools/code_stats``).

Fast and data-free: every pipeline file is parsed, never imported, so this runs
inside the API. Results are cached per file (path, size, mtime) and per
parent/child pair, so a page load after the first only re-reads changed files.

mle-claude runs are left to the operator analysis: their pipelines are skrub
plans over shared modules, which the operator DAG describes exactly.
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from code_stats import analyze_source, compare, normalized
from code_stats.diff import REPORTED

from .registry import REPO_ROOT, Run

_STDLIB = set(sys.stdlib_module_names)


def _stamp(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return str(path), st.st_size, st.st_mtime_ns


@lru_cache(maxsize=4096)
def _features(stamp: tuple[str, int, int]) -> dict:
    return analyze_source(Path(stamp[0]).read_text(errors="replace"))


@lru_cache(maxsize=4096)
def _normalized(stamp: tuple[str, int, int]) -> str | None:
    return normalized(Path(stamp[0]).read_text(errors="replace"))


@lru_cache(maxsize=4096)
def _diff(parent: tuple[str, int, int], child: tuple[str, int, int]) -> dict:
    a, b = Path(parent[0]).read_text(errors="replace"), Path(child[0]).read_text(errors="replace")
    return compare(_features(parent), _features(child), a, b,
                   normalized_pair=(_normalized(parent), _normalized(child)))


def warm(runs: list[Run]) -> None:
    """Fill the caches for every run (the API does this in the background at
    start, so the first page load of a large run is not the one paying)."""
    for run in runs:
        try:
            run_code(run)
        except Exception:   # noqa: BLE001 - a warm-up never fails the API
            pass


def _summary(diff: dict) -> str:
    """One line for the steps table."""
    if diff["same_code"]:
        return "same code"
    bits = [f"+{c}" for c in diff["components_added"]]
    bits += [f"−{c}" for c in diff["components_removed"]]
    changed = diff["params_changed"]
    for p in changed[:3]:
        bits.append(f"{p['component']}.{p['param']} {_short(p['old'])}→{_short(p['new'])}")
    if len(changed) > 3:
        bits.append(f"{len(changed) - 3} more parameter(s)")
    return " · ".join(bits) or "no component or parameter change"


def _short(v) -> str:
    s = str(v)
    return s if len(s) <= 16 else s[:15] + "…"


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return p.name


def run_code(run: Run) -> dict:
    if run.agent in ("mle-claude", "nano-mle"):
        return {"covered_by": "operator analysis",
                "reason": f"{run.agent} pipelines are skrub plans; the operator "
                          "explorer and operator statistics describe them exactly."}
    if run.originals is None or not run.originals.files:
        return {"covered_by": None, "reason": "no original scripts in pipelines/"}

    files = run.originals.files
    order = [m for m in run.modules if m in files] or sorted(files)
    stamps = {m: _stamp(files[m]) for m in order}
    pipelines = [{"name": m, "file": _rel(files[m]), **_features(stamps[m])} for m in order]

    diffs = {}
    for step in run.steps:
        if step.module in stamps and step.parent in stamps:
            d = _diff(stamps[step.parent], stamps[step.module])
            diffs[step.module] = {**d, "parent": step.parent, "summary": _summary(d)}

    changed = [d["lines_changed"] for d in diffs.values()]
    change = None if not changed else {
        "n": len(changed), "same_code": sum(d["same_code"] for d in diffs.values()),
        "median": statistics.median(changed), "max": max(changed),
        "similarity": round(statistics.median(d["similarity"] for d in diffs.values()), 3),
        # every parent -> child edge's ratio of changed code lines, for the CDF
        "ratios": sorted(d["change_ratio"] for d in diffs.values() if d["change_ratio"] is not None),
    }
    return {"covered_by": None, "pipelines": pipelines, "diffs": diffs,
            "summary": {**_aggregate(pipelines), "change": change}}


def _aggregate(pipelines: list[dict]) -> dict:
    ok = [p for p in pipelines if p["ok"]]
    n = len(ok)

    def dist(values: list[int]) -> dict | None:
        if not values:
            return None
        return {"median": statistics.median(values), "min": min(values), "max": max(values)}

    # components: pipelines using each, and the first pipeline (in trajectory order) to use it
    comps: dict[str, dict] = {}
    for p in ok:
        for c in {(c["kind"], c["name"], c["lib"]) for c in p["components"]}:
            kind, name, lib = c
            e = comps.setdefault(f"{kind}:{name}", {"kind": kind, "name": name, "lib": lib,
                                                     "n": 0, "first": p["name"]})
            e["n"] += 1
    libs = Counter(m for p in ok for m in p["imports"] if m not in _STDLIB)
    reads = Counter(r["path"] or f"({r['func']}, path not literal)"
                    for p in ok for r in {(x["func"], x["path"]): x for x in p["data"]["reads"]}.values())
    pip = Counter(pkg for p in ok for pkg in p["other"]["pip_installs"])
    splitters = Counter()
    for p in ok:
        for c in p["components"]:
            if c["kind"] == "splitter":
                key = "n_splits" if "n_splits" in c["params"] else "test_size"
                k = c["params"].get(key)
                if isinstance(k, str) and k.startswith("="):
                    k = f"⟨{k[1:]}⟩"          # not a literal: the expression, marked
                splitters[c["name"] + (f" ({key}={k})" if k is not None else "")] += 1
    pandas = Counter()
    polars = Counter()
    for p in ok:
        pandas.update(p["data"]["pandas"])
        polars.update(p["data"]["polars"])
    s = lambda key: [p["structure"][key] for p in ok]
    return {
        "n": len(pipelines), "n_failed": len(pipelines) - n,
        "failed": [{"name": p["name"], "error": p["error"]} for p in pipelines if not p["ok"]],
        "loc": dist([p["size"]["loc"] for p in ok]),
        "complexity": dist(s("complexity")),
        "functions": dist(s("functions")), "classes": dist(s("classes")),
        "loops": dist([p["structure"]["for"] + p["structure"]["while"] for p in ok]),
        "ifs": dist(s("if")), "tries": dist(s("try")),
        "column_writes": dist([p["data"]["column_writes"] for p in ok]),
        "components": sorted(comps.values(), key=lambda e: (_KIND_ORDER.get(e["kind"], 99), -e["n"], e["name"])),
        "libraries": [{"name": k, "n": v} for k, v in libs.most_common()],
        "reads": [{"path": k, "n": v} for k, v in reads.most_common()],
        "pip_installs": [{"name": k, "n": v} for k, v in pip.most_common()],
        "splitters": [{"name": k, "n": v} for k, v in splitters.most_common()],
        "pandas": dict(pandas.most_common()), "polars": dict(polars.most_common()),
        "gpu": sum(p["other"]["gpu"] for p in ok),
        "shell": sum(1 for p in ok if p["other"]["shell_calls"]),
        "inplace": sum(1 for p in ok if p["data"]["inplace"]),
    }


_KIND_ORDER = {k: i for i, k in enumerate((*REPORTED, "optimizer", "scheduler", "loss", "training",
                                           "data", "layer", "other"))}
