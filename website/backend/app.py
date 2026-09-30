"""JSON API over the corpus registry, consumed by the Next.js frontend.

Runs are addressed only by the ids the registry itself produced: a request
names ``<dataset>/<run>`` and is looked up in the registry, never joined onto a
filesystem path, so no request can reach a file outside the corpus.

The API has no login of its own. It listens on a Unix socket in a directory
only the owner can open (see ``__main__``), and the frontend -- which does the
authentication -- is its only client.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Response

from . import actions, jobs
from .registry import (REPO_ROOT, Dataset, Metric, Run, RuntimeStore, Source, StepInfo,
                       current_stratum_commit, load_corpus)
from .runtime_profile import build_profile
from .tree import build_tree

app = FastAPI(title="mle-trajectories", docs_url=None, redoc_url=None, openapi_url=None)


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _metric(m: Metric) -> dict:
    return {"name": m.name, "lower_is_better": m.lower_is_better}


def _source(run: Run, s: Source) -> dict:
    have, total = run.coverage(s)
    return {"name": s.name, "label": s.label, "skrub": s.skrub, "dirs": s.rel_dirs(run.path),
            "default": s is run.default_source(), "hidden": s.hidden,
            "fold_identical_code": s.fold_identical_code, "note": s.note,
            "n_files": len(s.files), "coverage": [have, total]}


def _runtime(r: RuntimeStore) -> dict:
    return {"name": r.name, "file": r.path.name, "source": r.source,
            "sample_rows": r.sample_rows, "data": r.data, "label": r.label, "note": r.note,
            "hidden": r.hidden, "measured_at": r.measured_at,
            "n_ok": r.n_ok, "n_failed": r.n_failed, "n_code_changed": r.n_code_changed,
            "n_old_build": r.n_old_build, "commits": r.commits}


def _step(s: StepInfo) -> dict:
    return {"module": s.module, "parent": s.parent, "phase": s.phase,
            "score": s.score, "desc": s.desc}


def _run_summary(run: Run) -> dict:
    best = run.best()
    src = run.default_source()
    return {
        "id": run.id, "dataset": run.dataset.name, "name": run.name, "label": run.label,
        "agent": run.agent, "metric": _metric(run.metric), "lineage": run.lineage_from,
        "n_steps": len(run.steps),
        "best": None if best is None else {"module": best.module, "score": best.score},
        "default_source": None if src is None else _source(run, src),
        "n_sources": len([s for s in run.sources if not s.hidden]),
        "runtime": [_runtime(r) for r in run.runtime],
        "n_warnings": len(run.warnings),
    }


def _dataset(ds: Dataset, runs: bool = True) -> dict:
    out = {"name": ds.name, "label": ds.label, "task": ds.task, "data": ds.data,
           "data_status": ds.data_status(), "samples": ds.samples, "note": ds.note,
           "metric": _metric(ds.metric), "warnings": ds.warnings}
    if runs:
        out["runs"] = [_run_summary(r) for r in ds.runs]
    return out


_CORPUS_TTL_S = 5.0
_corpus_cache: tuple[float, list[Dataset]] | None = None
_corpus_lock = threading.Lock()


def _corpus() -> list[Dataset]:
    """The registry, re-read at most every few seconds: one page load asks for a
    run several times (detail, tree, analysis, runtime) and re-parsing every
    manifest and trajectory each time costs ~0.4 s per request."""
    global _corpus_cache
    with _corpus_lock:
        now = time.monotonic()
        if _corpus_cache is None or now - _corpus_cache[0] > _CORPUS_TTL_S:
            _corpus_cache = (now, load_corpus())
        return _corpus_cache[1]


def _find_run(dataset: str, run: str) -> Run:
    for ds in _corpus():
        if ds.name == dataset:
            for r in ds.runs:
                if r.name == run:
                    return r
    raise HTTPException(404, f"no run {dataset}/{run}")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/corpus")
def corpus() -> dict:
    return {"stratum_commit": current_stratum_commit(),
            "datasets": [_dataset(ds) for ds in _corpus()]}


def _source_or_404(r: Run, name: str | None) -> Source:
    src, _ = r.pick_source(name)
    if src is None or (name and src.name != name):
        raise HTTPException(404, f"run {r.id} has no pipeline source {name!r}")
    return src


@app.get("/api/runs/{dataset}/{run}")
def run_detail(dataset: str, run: str, source: str | None = None,
               runtime: str | None = None) -> dict:
    r = _find_run(dataset, run)
    src, src_why = r.pick_source(source)
    rt, rt_why = r.pick_runtime(runtime, src)
    return {
        "selection": {
            "source": None if src is None else {"name": src.name, "reason": src_why},
            "runtime": None if rt is None else {"name": rt.name, "reason": rt_why},
        },
        **_run_summary(r),
        "note": r.note,
        "path": _rel(r.path),
        "dataset_info": _dataset(r.dataset, runs=False),
        "stratum_commit": current_stratum_commit(),
        "trajectory": None if r.trajectory_file is None else {
            "file": r.trajectory_file.name,
            "meta": {k: str(v) for k, v in r.trajectory_meta.items()}},
        "originals": None if r.originals is None else _source(r, r.originals),
        "sources": [_source(r, s) for s in r.sources],
        "steps": [_step(s) for s in r.steps],
        "facts": r.facts(),
        "warnings": r.warnings,
    }


@app.get("/api/runs/{dataset}/{run}/tree")
def run_tree(dataset: str, run: str, source: str | None = None) -> dict:
    r = _find_run(dataset, run)
    src, _ = r.pick_source(source)
    return build_tree(r, src)


@app.get("/api/runs/{dataset}/{run}/runtime/{store}")
def run_runtime(dataset: str, run: str, store: str) -> dict:
    r = _find_run(dataset, run)
    rs = next((x for x in r.runtime if x.name == store), None)
    if rs is None:
        raise HTTPException(404, f"run {r.id} has no runtime store {store!r}")
    known: set[str] = set()
    src = next((s for s in r.sources if s.name == rs.source), None)
    cached = jobs.cached_result(r, src) if src else None
    if cached:   # physical op classes restore the "Op" suffix in measured labels
        known = set(json.loads(cached.read_text())["stats"].get("physical_classes") or ())
    return build_profile(r, rs, current_stratum_commit(), known)


@app.get("/api/runs/{dataset}/{run}/analysis/{source}")
def run_analysis(dataset: str, run: str, source: str, start: bool = False,
                 retry: bool = False) -> Response:
    """Status of the operator analysis; its JSON inlined when ready. ``start``
    builds it in the background when missing (``retry`` also after a failure)."""
    r = _find_run(dataset, run)
    src = _source_or_404(r, source)
    st = jobs.start(r, src, retry=retry) if (start or retry) else jobs.status(r, src)
    if st["status"] == "ready":
        # the result is already JSON on disk: splice it in instead of re-parsing
        body = '{"status":"ready","data":' + Path(st["path"]).read_text() + "}"
        return Response(body, media_type="application/json")
    return Response(json.dumps({k: v for k, v in st.items() if k != "path"}),
                    media_type="application/json")


# --------------------------------------------------------------------------- #
# actions: the frontend sends parameters, actions.py builds and validates the
# command from the registry run
# --------------------------------------------------------------------------- #
def _action(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except actions.ActionError as exc:
        raise HTTPException(400, str(exc)) from None


@app.get("/api/runs/{dataset}/{run}/actions")
def run_actions(dataset: str, run: str) -> dict:
    r = _find_run(dataset, run)
    return {
        "sweep": {
            "sources": [{"name": s.name, "label": s.label, "coverage": list(r.coverage(s)),
                         "default": s is r.default_source()}
                        for s in r.sources if s.skrub and s.files and not s.hidden],
            "data": actions.data_options(r),
        },
        "jobs": actions.jobs_for(r),
        # a sweep anywhere on the node blocks starting another one
        "busy": next(({"id": j["id"], "run": j["run"], "label": j["label"]}
                      for j in actions.running("runtime-sweep")), None),
    }


@app.post("/api/runs/{dataset}/{run}/actions/runtime-sweep/plan")
def sweep_plan(dataset: str, run: str, params: dict = Body(...), listing: bool = False) -> dict:
    return _action(actions.plan_sweep, _find_run(dataset, run), params, listing=listing)


@app.post("/api/runs/{dataset}/{run}/actions/runtime-sweep")
def sweep_start(dataset: str, run: str, params: dict = Body(...)) -> dict:
    return _action(actions.start_sweep, _find_run(dataset, run), params)


@app.get("/api/jobs")
def jobs_list() -> dict:
    return {"jobs": actions.jobs_for(limit=20)}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    return _action(actions.job_status, job_id, log_lines=60)


@app.post("/api/jobs/{job_id}/stop")
def job_stop(job_id: str) -> dict:
    return _action(actions.stop, job_id)
