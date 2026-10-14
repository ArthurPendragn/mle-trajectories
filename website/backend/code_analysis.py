"""Static code analysis of a run's original scripts (``tools/code_stats``).

Fast and data-free: every pipeline file is parsed, never imported, so this runs
inside the API. Results are cached per file (path, size, mtime) and per
parent/child pair, so a page load after the first only re-reads changed files.

What a script imports from the run's own modules (mle-claude's ``common.py``,
``features.py``, ...) is inlined before measuring (``code_stats.inline``), so a
five-line script over shared helpers is measured with the code it uses.
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from code_stats import analyze_source, compare, inline_local, normalized
from code_stats.diff import REPORTED

from .registry import REPO_ROOT, Run

_STDLIB = set(sys.stdlib_module_names)


def _stamp(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return str(path), st.st_size, st.st_mtime_ns


# A script is keyed by its own stamp and the stamps of its run's other files:
# what it imports from sibling modules (common.py, features.py, ...) is inlined
# before measuring, so an edit there invalidates it too.
Key = tuple[tuple[str, int, int], tuple[tuple[str, int, int], ...]]


@lru_cache(maxsize=4096)
def _source(key: Key) -> tuple[str, dict[str, list[str]]]:
    own = Path(key[0][0]).read_text(errors="replace")
    mods = {Path(s[0]).stem: Path(s[0]).read_text(errors="replace") for s in key[1]}
    return inline_local(own, mods) if mods else (own, {})


@lru_cache(maxsize=4096)
def _features(key: Key) -> dict:
    text, inlined = _source(key)
    f = analyze_source(text)
    if inlined:
        own = analyze_source(Path(key[0][0]).read_text(errors="replace"))
        f["inlined"] = {"modules": inlined, "own_loc": own["size"]["loc"]}
    return f


@lru_cache(maxsize=4096)
def _normalized(key: Key) -> str | None:
    return normalized(_source(key)[0])


@lru_cache(maxsize=4096)
def _diff(parent: Key, child: Key) -> dict:
    return compare(_features(parent), _features(child), _source(parent)[0], _source(child)[0],
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
    if run.originals is None or not run.originals.files:
        return {"covered_by": None, "reason": "no original scripts in pipelines/"}

    files = run.originals.files
    order = [m for m in run.modules if m in files] or sorted(files)
    every = {m: _stamp(p) for m, p in files.items()}
    stamps = {m: (every[m], tuple(v for k, v in sorted(every.items()) if k != m)) for m in order}
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


CUSTOM = "⟨self-defined network⟩"
# scripts with no recognised model: hand-written scoring rules, heuristics
NO_MODEL = "⟨unknown / custom⟩"

# one entity per model regardless of task variant or API: LGBMClassifier,
# LGBMRegressor and lightgbm.train are all LightGBM. Kernels stay apart
# (HistGradientBoosting is not LightGBM). Unlisted names fall back to their
# variant suffix stripped, so a new XyzClassifier still merges with XyzRegressor.
MODEL_ENTITY = {
    "lightgbm.train": "LightGBM", "LGBMClassifier": "LightGBM", "LGBMRegressor": "LightGBM",
    "LGBMRanker": "LightGBM",
    "xgboost.train": "XGBoost", "XGBClassifier": "XGBoost", "XGBRegressor": "XGBoost",
    "XGBRanker": "XGBoost",
    "CatBoostClassifier": "CatBoost", "CatBoostRegressor": "CatBoost", "CatBoostRanker": "CatBoost",
}
_VARIANT = re.compile(r"(Classifier|Regressor|Ranker)$")


def model_entity(name: str) -> str:
    """The model ``name`` stands for, merged across classification/regression
    variants and sklearn-wrapper vs native API."""
    return MODEL_ENTITY.get(name) or _VARIANT.sub("", name) or name


def run_dashboard(run: Run) -> dict | None:
    """One run's code analysis reduced for the cross-run dashboard: per step its
    size, change against the parent and score, and per component kind how many
    of its scripts use each name. ``None`` for runs the analysis does not cover."""
    code = run_code(run)
    if code.get("covered_by") is not None or not code.get("pipelines"):
        return None
    scores = {s.module: s.score for s in run.steps if s.module}
    best = run.best()
    steps = []
    for p in code["pipelines"]:
        d = code["diffs"].get(p["name"])
        steps.append({
            "name": p["name"], "ok": p["ok"], "loc": p["size"]["loc"],
            "own_loc": p["inlined"]["own_loc"] if "inlined" in p else None,
            "cx": p["structure"]["complexity"] if p["ok"] else None,
            "score": scores.get(p["name"]),
            "changed": None if d is None else d["lines_changed"],
            "ratio": None if d is None else d["change_ratio"],
            "same": None if d is None else d["same_code"],
            "params": None if d is None else len(d["params_changed"]),
            "comps": None if d is None else len(d["components_added"]) + len(d["components_removed"]),
        })
    ok = [p for p in code["pipelines"] if p["ok"]]
    # networks the scripts define themselves, folded into one name ("folded") or
    # models grouped by their library ("family"); a script counts once per name
    custom = {c["name"] for p in ok for c in p["components"] if c.get("network")}
    usage: dict[str, Counter] = {}
    for p in ok:
        used = {(c["kind"], c["name"], c["lib"]) for c in p["components"] if c["kind"] in REPORTED}
        for kind, name, _ in {(k, n, None) for k, n, _ in used}:
            usage.setdefault(kind, Counter())[name] += 1
        models = [(n, lib) for k, n, lib in used if k == "model"]
        if not models:
            for key in ("model", "model_folded", "model_merged", "model_folded_merged", "family"):
                usage.setdefault(key, Counter())[NO_MODEL] += 1
        for name in {CUSTOM if n in custom else n for n, _ in models}:
            usage.setdefault("model_folded", Counter())[name] += 1
        # merged after naming, so a script using two variants counts once
        for name in {model_entity(n) for n, _ in models}:
            usage.setdefault("model_merged", Counter())[name] += 1
        for name in {CUSTOM if n in custom else model_entity(n) for n, _ in models}:
            usage.setdefault("model_folded_merged", Counter())[name] += 1
        for fam in {"custom network" if n in custom else (lib or "?") for n, lib in models}:
            usage.setdefault("family", Counter())[fam] += 1
    s = code["summary"]
    return {
        "id": run.id, "dataset": run.dataset.name, "dataset_label": run.dataset.label,
        "run": run.name, "label": run.label,
        # the variant is its own agent here: meta-mle's policies search differently
        "agent": f"{run.agent} · {run.variant}" if run.agent and run.variant else run.agent,
        "metric": {"name": run.metric.name, "lower_is_better": run.metric.lower_is_better},
        "best": None if best is None else best.module,
        "n": len(code["pipelines"]), "n_ok": len(ok), "steps": steps,
        "usage": {k: dict(v.most_common()) for k, v in usage.items()},
        "libraries": {l["name"]: l["n"] for l in s["libraries"]},
        "metrics": dict(Counter(m for p in ok for m in set(p["metrics"])).most_common()),
        "gpu": s["gpu"], "shell": s["shell"], "pip": len(s["pip_installs"]) > 0,
    }
