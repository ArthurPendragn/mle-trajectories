"""Child process: run ONE pipeline under stratum's scheduler and dump its stats.

    python -m pipeline_analyzer._measure <pipeline.py> <out.json>

Run with the CWD the pipeline's relative paths resolve against (the folder
holding ``input/``). ``runtime.py`` is the parent that drives this; it lives in
its own process so one pipeline's crash, memory blow-up or hang cannot take the
batch with it, and so peak RSS is attributable to a single pipeline.

The pipeline file is executed as written (``run_name="__main__"``, so its scoring
block runs), with three interventions -- none of them touches how data is read;
a smaller sweep runs on a sample folder (``tools/dataset_sample``) instead:

1. ``make_grid_search`` is replaced, so the scoring call goes straight to
   ``stratum._api.grid_search`` (what stratum's own patch does under
   ``scheduler=True``) and we can time it and keep the scheduler for its stats.
   ``cv`` is left alone: ``grid_search`` resolves the splitter declared on the
   plan via ``mark_as_X(cv=..., split_kwargs=...)`` itself. We hand back a shim
   whose ``results_`` looks like skrub's pandas frame -- stratum's is a polars
   frame keyed ``id``/``scores`` -- so each file's own reporting block runs and
   no pipeline needs editing.
2. A file that only *defines* a plan (module-level ``pred``, the mle-claude
   convention: its harness did the scoring) is scored after it ran, the way that
   harness did -- ``pred.skb.make_grid_search(fitted=True, refit=False)`` with no
   ``cv=`` and ``--scoring`` -- through the same replaced ``make_grid_search``.
3. The file runs under ``skrub.config_context(eager_data_ops=False)``, so
   building the plan computes no previews: without it every recorded op is
   evaluated once at build time, which on a data-lake plan re-reads every table
   before the scored run starts. Only ``total_s`` and the memory curve before
   ``scored_from_s`` change; the scored ``wall_s`` does not.

Memory is sampled by ``memory_tracker.MemoryTracker``, a side-car process
polling RSS every 100 ms, which gives the shape of the run and not just its
high-water mark. Two peaks are reported and they answer different questions:
``memory.peak_mb`` is the sampled curve's maximum for *this* interpreter, while
``max_rss_mb`` comes from ``getrusage`` and covers this process plus any worker
processes an estimator forked, so it is the number to trust for "how much did
this pipeline need". A short spike between two polls shows up in the second but
not the first.

``stats=True`` is set only around the scoring call, and only here -- it makes
stratum time every operator, which is exactly what we want to collect but is
overhead the library should not carry by default.
"""
from __future__ import annotations

import argparse
import json
import resource
import runpy
import time
import traceback
from collections import defaultdict
from pathlib import Path


def _declared_cv(dag):
    """The splitter a plan declares through ``mark_as_X(cv=...)``, for the record.

    Reporting only -- ``stratum._api.grid_search`` resolves the plan's splitter
    itself (including a ``cv`` that is a DataOp), so this exists to write which
    splitter drove the folds into the stats, not to influence them. Uses
    stratum's fast DFS rather than skrub's traversal, which is quadratic on plans
    with many shared nodes.
    """
    from skrub._data_ops._data_ops import SplitX
    from stratum.frontend._skrub_graph import build_graph

    for node in build_graph(dag)["nodes"].values():
        impl = node._skrub_impl
        if isinstance(impl, SplitX):
            return impl.cv
    return None


def _aggregate(timings) -> list[dict]:
    """``[(op, seconds), ...]`` -> one row per operator, slowest first."""
    total_s, counts = defaultdict(float), defaultdict(int)
    for op, seconds in timings or ():
        total_s[op] += seconds
        counts[op] += 1
    rows = [{"op": op, "count": counts[op], "time_s": round(total_s[op], 6)}
            for op in total_s]
    rows.sort(key=lambda r: -r["time_s"])
    return rows


def _memory_summary(samples, t0: float, *, interval_s: float,
                    marks: dict, max_points: int = 120) -> dict:
    """Reduce an RSS sample series to what a report can use.

    Keeps a downsampled curve (so the store stays small -- the full series goes
    to CSV when the runner asks for one) plus the summary statistics, with times
    relative to the start of tracking.
    """
    if not samples:
        return {}
    series = [(round(ts - t0, 3), round(mb, 1)) for ts, mb in samples]
    values = [mb for _, mb in series]
    step = max(1, len(series) // max_points)
    curve = series[::step]
    if curve[-1] != series[-1]:
        curve.append(series[-1])
    peak_at, peak = max(series, key=lambda s: s[1])[0], max(values)
    return {
        "peak_mb": round(peak, 1),
        "peak_at_s": peak_at,
        "mean_mb": round(sum(values) / len(values), 1),
        "start_mb": series[0][1],
        "end_mb": series[-1][1],
        "n_samples": len(series),
        "interval_s": interval_s,
        "duration_s": series[-1][0],
        "curve": curve,
        **marks,
    }


def _pool_stats(pool) -> dict:
    s = getattr(pool, "stats", None)
    if s is None:
        return {}
    return {
        "hits": getattr(s, "hits", None),
        "misses": getattr(s, "misses", None),
        "hit_rate": round(getattr(s, "hit_rate", 0.0), 4),
        "evictions": getattr(s, "evictions", None),
        "serialize_s": round(getattr(s, "serialize_time", 0.0), 6),
        "deserialize_s": round(getattr(s, "deserialize_time", 0.0), 6),
        "bytes_spilled": getattr(s, "bytes_spilled", None),
        "bytes_loaded": getattr(s, "bytes_loaded", None),
    }


def _scoring_honoured(scoring) -> bool:
    """Whether stratum will actually use this ``scoring``.

    ``get_scoring_func`` dispatches on ``type(scoring)`` being ``str`` or
    ``_Scorer`` and otherwise falls back to ``mean_squared_error`` -- so a plain
    callable scorer, or ``None``, is silently discarded and the reported number is
    an MSE with the sort order flipped (deem-data/stratum#200). Recording this
    keeps such a score from being read as the metric the pipeline asked for.

    Only a bare string is reported as honoured: a ``make_scorer`` scorer reaches
    its metric but loses the scorer's kwargs, and a ``neg_*`` scorer's sign is
    not applied to the value, so neither round-trips to sklearn's number either.
    """
    if isinstance(scoring, str):
        return True
    if scoring is None:
        return False
    return False


def _scores(results) -> list[float]:
    """Scores out of stratum's polars results frame, best first."""
    try:
        return [float(v) for v in results["scores"].to_list()]
    except Exception:  # noqa: BLE001 - a score list is a nicety, not the point
        return []


def measure(path: Path, *, stats: bool,
            mem_mode: str = "process", mem_interval: float = 0.1,
            mem_csv: Path | None = None, scoring: str | None = None) -> dict:
    import pandas as pd
    import skrub
    import stratum
    from skrub._data_ops._skrub_namespace import SkrubNamespace
    from stratum._api import grid_search as stratum_grid_search

    tracker = None
    if mem_mode != "off":
        from .memory_tracker import MemoryTracker
        if mem_csv:
            # This process may have a different cwd than whoever asked for the
            # path, and the tracker's side-car cannot create it.
            mem_csv = Path(mem_csv).resolve()
            mem_csv.parent.mkdir(parents=True, exist_ok=True)
        tracker = MemoryTracker(mode=mem_mode, interval_sec=mem_interval,
                                live_dump_path=str(mem_csv) if mem_csv else None)

    out: dict = {"status": "error", "error": "the pipeline never scored a plan "
                 "(no make_grid_search call)"}
    marks: dict = {}

    def make_grid_search(self, **kwargs):
        dag = self._data_op
        # Precedence matches grid_search's own: an explicit cv wins, else the plan's.
        explicit_cv = kwargs.get("cv")
        cv = explicit_cv if explicit_cv is not None else _declared_cv(dag)

        t0 = time.perf_counter()
        with stratum.config(scheduler=True, stats=stats, debug_graph=False,
                            open_graph=False):
            sched = stratum_grid_search(dag=dag, cv=explicit_cv,
                                        scoring=kwargs.get("scoring"))
        wall = time.perf_counter() - t0
        if tracker is not None:
            # Where the scored run sits inside the memory curve, which also
            # covers plan construction before it.
            marks["scored_from_s"] = round(t0 - tracker.t0, 3)
            marks["scored_to_s"] = round(t0 + wall - tracker.t0, 3)

        ops = _aggregate(sched.timings)
        scores = _scores(sched.results_)
        out.update(
            status="ok", error=None,
            wall_s=round(wall, 4),
            op_time_s=round(sum(r["time_s"] for r in ops), 4),
            buffer_overhead_s=round(sched.buffer_pool_overhead, 4),
            n_op_calls=len(sched.timings or ()),
            ops=ops,
            pool=_pool_stats(sched.pool),
            scores=scores,
            best_score=scores[0] if scores else None,
            cv=repr(cv) if cv is not None else None,
            cv_declared_on_plan=explicit_cv is None and cv is not None,
            scoring=kwargs.get("scoring") if isinstance(kwargs.get("scoring"), str)
                    else getattr(kwargs.get("scoring"), "__name__", None),
            scoring_honoured=_scoring_honoured(kwargs.get("scoring")),
        )

        # Hand back something shaped like skrub's search so the pipeline's own
        # reporting block (results_["mean_test_score"]) runs unchanged.
        frame = pd.DataFrame({"mean_test_score": scores,
                              "variant": list(sched.results_["id"])})

        class _Search:
            results_ = frame
            scheduler_ = sched

        return _Search()

    SkrubNamespace.make_grid_search = make_grid_search

    if tracker is not None:
        tracker.start()
    t0 = time.perf_counter()
    try:
        with skrub.config_context(eager_data_ops=False):
            module = runpy.run_path(str(path), run_name="__main__")
            pred = module.get("pred")
            if out["status"] != "ok" and hasattr(pred, "skb"):
                # A plan-only file: score it as its harness did.
                out["scored_by"] = "runner (module-level pred)"
                pred.skb.make_grid_search(n_jobs=1, fitted=True, refit=False,
                                          scoring=scoring)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        detail = f"{type(exc).__name__}: {exc}"
        if out["status"] == "ok":
            # The plan was scored and timed; only what the file did afterwards
            # (its own printing, a submission write) blew up. Keep the numbers
            # rather than making an expensive pipeline look unmeasured.
            out["post_error"] = detail
        else:
            out["status"] = "error"
            out["error"] = detail
            out["traceback"] = traceback.format_exc(limit=6)
    out["total_s"] = round(time.perf_counter() - t0, 4)
    if tracker is not None:
        samples = tracker.stop()
        out["memory"] = _memory_summary(samples, tracker.t0,
                                        interval_s=mem_interval, marks=marks)
        out["mem_mode"] = mem_mode
        if mem_csv:
            # The live dump carries raw perf_counter stamps (it exists so a killed
            # run still leaves a trace); rewrite it with times relative to the
            # start of tracking now that the run is over. Never fatal: a run that
            # took minutes must not be thrown away over a side file.
            try:
                tracker.write_csv(str(mem_csv))
            except OSError as exc:
                out["mem_csv_error"] = f"{type(exc).__name__}: {exc}"
    # Estimators fan out over threads (sharedmem) or processes (loky), so take
    # whichever of the two peaks is higher rather than only this interpreter's.
    out["max_rss_mb"] = round(max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024, 1)
    out["stats_enabled"] = stats
    out["eager_previews"] = False
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline_analyzer._measure",
                                 description=__doc__)
    ap.add_argument("pipeline", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--no-stats", action="store_true")
    ap.add_argument("--scoring", default=None,
                    help="scorer for a file that only defines `pred`")
    ap.add_argument("--mem-mode", default="process",
                    choices=("process", "system", "off"))
    ap.add_argument("--mem-interval", type=float, default=0.1)
    ap.add_argument("--mem-csv", type=Path, default=None,
                    help="write the full RSS sample series here as it is collected")
    args = ap.parse_args(argv)

    result = measure(args.pipeline, stats=not args.no_stats, mem_mode=args.mem_mode,
                     mem_interval=args.mem_interval, mem_csv=args.mem_csv,
                     scoring=args.scoring)
    args.out.write_text(json.dumps(result))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
