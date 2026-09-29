"""Build one run's operator analysis for one pipeline source, in its own process.

    python -m website.backend.worker <dataset>/<run> <source> <out.json>

Loading a pipeline imports it (skrub, stratum, torch, the run's own modules), so
this never runs inside the API process: the API starts it (``jobs.py``) and
serves the JSON it leaves behind. Output:

* ``merged``  -- the union operator DAG, in the payload format the explorer reads
  (``pipeline_analyzer.merged``),
* ``stats``   -- operator statistics at the logical and physical altitude,
* ``failed``  -- pipelines whose plan could not be extracted, with the error,
* ``folded``  -- code-identical steps that share one DAG (``fold_identical_code``).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from .registry import current_stratum_commit, load_corpus


def _stats(per_pipe: list[dict]) -> dict:
    """Per operator type: total, pipelines containing it, and the per-pipeline
    count distribution over *all* pipelines (a pipeline lacking it counts 0) --
    the same table as the static report's ``_stats_block``."""
    n = len(per_pipe)
    if n == 0:
        return {"n_pipelines": 0, "rows": [], "sizes": None}
    rows = []
    for op in set().union(*per_pipe):
        counts = [h.get(op, 0) for h in per_pipe]
        rows.append({
            "op": op, "total": sum(counts), "present": sum(1 for c in counts if c),
            "mean": statistics.mean(counts), "median": statistics.median(counts),
            "std": statistics.pstdev(counts) if n > 1 else 0.0,
            "min": min(counts), "max": max(counts),
        })
    rows.sort(key=lambda r: (-r["total"], r["op"]))
    sizes = [sum(h.values()) for h in per_pipe]
    return {"n_pipelines": n, "rows": rows,
            "sizes": {"total": sum(sizes), "median": statistics.median(sizes),
                      "min": min(sizes), "max": max(sizes)}}


def _skip_creation_stacks() -> None:
    """Stop skrub recording a formatted call stack for every DataOp it creates.

    skrub keeps it only to say "this node was defined here" in error messages,
    and it dominated plan building: on the nyc data-lake run, 4.9 of the 6.1 s
    spent building its largest plan (415 operators). Loading is 2.5x faster
    without it and the extracted DAGs are byte-identical (signatures, histograms,
    physical plans; checked on five pipelines of that run). skrub already treats
    a missing stack as normal (``_creation_stack_lines = None``).
    """
    import skrub._data_ops._data_ops as data_ops
    data_ops._format_data_op_creation_stack = lambda: None


def build(run_id: str, source_name: str) -> dict:
    _skip_creation_stacks()
    from pipeline_analyzer.lineage import (build_lineage, build_lineage_from_trajectory,
                                           fold_translation_variants)
    from pipeline_analyzer.loader import load_all
    from pipeline_analyzer.merged import build_merged
    from pipeline_analyzer.trajectory import parse

    run = next((r for ds in load_corpus() for r in ds.runs if r.id == run_id), None)
    if run is None:
        raise SystemExit(f"no run {run_id}")
    source = next((s for s in run.sources if s.name == source_name), None)
    if source is None:
        raise SystemExit(f"run {run_id} has no source {source_name}")

    # pipelines resolve ./input and sibling modules relative to the run
    os.chdir(run.path)
    folded: list = []
    if run.trajectory_file:
        traj = parse(json.loads(run.trajectory_file.read_text()), run.path)
        pipelines = load_all(source.dirs, names=list(traj.by_module()))
        if source.fold_identical_code:
            folded = fold_translation_variants(pipelines, traj)
        lineage = build_lineage_from_trajectory(pipelines, traj)
    else:
        # mle-claude: results.json names the pipelines (not a file pattern, so
        # ablation_*.py count and scratch files do not)
        pipelines = load_all(source.dirs, names=run.modules)
        lineage = build_lineage(pipelines, results_path=source.dirs[0] / "results.json")
        lineage.lower_is_better = bool(run.metric.lower_is_better)

    ok = [p for p in pipelines if p.ok]
    phys = [p for p in ok if p.phys_dag is not None]
    merged = build_merged(lineage)
    return {
        "version": 1,
        "run": run_id,
        "source": source_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stratum_commit": current_stratum_commit(),
        "n_pipelines": len(pipelines),
        "merged": json.loads(merged.to_json()),
        "stats": {
            "logical": _stats([p.dag.histogram() for p in ok]),
            "physical": _stats([p.phys_dag.histogram(specific=True) for p in phys]),
            "physical_missing": len(ok) - len(phys),
            "physical_classes": sorted({n.op_type for p in phys for n in p.phys_dag.nodes.values()}),
        },
        "failed": [{"name": p.name, "error": (p.error or "").strip()}
                   for p in pipelines if not p.ok],
        "folded": folded,
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    run_id, source, out = argv[0], argv[1], Path(argv[2]).resolve()
    t0 = time.time()
    print(f"analyzing {run_id} [{source}] ...", flush=True)
    result = build(run_id, source)
    result["elapsed_s"] = round(time.time() - t0, 1)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, separators=(",", ":")))
    os.replace(tmp, out)
    print(f"{result['n_pipelines'] - len(result['failed'])}/{result['n_pipelines']} "
          f"extracted in {result['elapsed_s']}s -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
