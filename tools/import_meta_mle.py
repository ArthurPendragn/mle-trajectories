"""Import a nano-mle workspace as a run folder in the mle-claude layout.

nano-mle writes skrub plans itself, so its plans are the originals:

    <dataset>/<run>/
        run.toml                 agent = "nano-mle"
        task_description.txt     the task text the agent was given
        workspace.json, report.md, graph.json   nano-mle's own exports (journal)
        pipelines/
            pipeline_NN.py       one per scored expansion, its grid included
            results.json         lineage, best-first (same shape as mle-claude's)
            evaluation_setup.py  the locked X/y/CV/scorer plan
            data_exploration_N.py, probe_N.py   evidence, not pipelines

Plans are copied byte for byte. They read the task sources by the absolute paths
they had during the run.

Usage:
    python tools/import_nano_mle.py WORKSPACE DATASET_DIR RUN_NAME [--label LABEL]
"""

import argparse
import json
import shutil
import sqlite3
from pathlib import Path


def records(db, kind):
    return [json.loads(p) for (p,) in db.execute(
        "select payload from records where kind = ? order by seq", (kind,))]


def meta(db, key):
    row = db.execute("select payload from meta where key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def last_ok_plan(workspace, record):
    """The plan of the record's successful attempt, else its last attempt."""
    attempts = [workspace / "artifacts" / record["id"] / a for a in record.get("attempt_ids", [])]
    for attempt in reversed(attempts):
        response = attempt / "response.json"
        if response.is_file() and json.loads(response.read_text()).get("status") == "ok":
            return attempt
    return attempts[-1] if attempts else None


def wall_time(attempt):
    try:
        return json.loads((attempt / "response.json").read_text()).get("wall_s")
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workspace", type=Path, help="nano-mle workspace (the directory holding state.db)")
    parser.add_argument("dataset", type=Path, help="dataset folder in this repo")
    parser.add_argument("name", help="run folder name, e.g. nano_mle_sol_run_1")
    parser.add_argument("--label")
    parser.add_argument("--force", action="store_true", help="replace an existing run folder")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    db = sqlite3.connect(f"file:{workspace / 'state.db'}?mode=ro", uri=True)
    out = args.dataset.resolve() / args.name
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    pipelines = out / "pipelines"
    pipelines.mkdir(parents=True)

    contract = meta(db, "contract") or {}
    spec = contract.get("spec", {})
    candidates = {c["id"]: c for c in records(db, "candidate")}
    expansions = [e for e in records(db, "expansion") if e.get("status") == "ok" and e.get("candidate_ids")]
    names = {e["id"]: f"pipeline_{i:02d}" for i, e in enumerate(expansions, 1)}

    rows = []
    for expansion in expansions:
        variants = [candidates[c] for c in expansion["candidate_ids"]
                    if c in candidates and candidates[c].get("score") is not None]
        if not variants:
            continue
        attempt = workspace / Path(variants[0]["source_path"]).parent
        shutil.copyfile(attempt / "plan.py", pipelines / f"{names[expansion['id']]}.py")
        parent = candidates.get(expansion["parent_id"], {}).get("batch_id")
        grid = sorted(({**v.get("configuration_description", {}), "mean_test_score": v["score"]}
                       for v in variants), key=lambda g: g["mean_test_score"], reverse=True)
        rows.append({"pipeline": names[expansion["id"]], "parent": names.get(parent),
                     "description": expansion["proposal"]["description"],
                     "metric": contract.get("scoring"), "cv": spec.get("cv"),
                     "score": max(v["score"] for v in variants),
                     "duration_s": wall_time(attempt),
                     "extra": {"grid": grid} if len(grid) > 1 else {},
                     "nano_mle": {"expansion_id": expansion["id"], "parent_candidate": expansion["parent_id"],
                                  "direction": expansion.get("direction"),
                                  "fold_scores": max(variants, key=lambda v: v["score"]).get("fold_scores")}})
    rows.sort(key=lambda r: r["score"], reverse=True)
    (pipelines / "results.json").write_text(json.dumps(rows, indent=1))

    evidence = {"evaluation_setup": [s for s in records(db, "evaluation_setup") if s.get("status") == "ok"][-1:],
                "data_exploration": [e for e in records(db, "exploration") if e.get("status") == "ok"],
                "probe": [p for p in records(db, "probe") if p.get("status") == "ok"]}
    for prefix, items in evidence.items():
        for i, item in enumerate(items, 1):
            attempt = last_ok_plan(workspace, item)
            name = prefix if prefix == "evaluation_setup" else f"{prefix}_{i}"
            if attempt and (attempt / "plan.py").is_file():
                shutil.copyfile(attempt / "plan.py", pipelines / f"{name}.py")

    for exported in ("workspace.json", "report.md", "graph.json"):
        if (workspace / exported).is_file():
            shutil.copyfile(workspace / exported, out / exported)
    task = meta(db, "task") or {}
    (out / "task_description.txt").write_text(task.get("description", ""))
    failed = len([e for e in records(db, "expansion") if e.get("status") != "ok"])
    label = args.label or f"nano-mle ({meta(db, 'model')})"
    note = (f"nano-mle {meta(db, 'policy')} search, model {meta(db, 'model')}. "
            f"{len(rows)} scored expansions, {failed} failed. Locked {contract.get('scoring')} on "
            f"{contract.get('rows')} rows. Plans read task sources by absolute path.")
    (out / "run.toml").write_text(f'agent = "nano-mle"\nlabel = {json.dumps(label)}\nnote = {json.dumps(note)}\n')
    print(f"{out}: {len(rows)} pipelines, best {rows[0]['score'] if rows else None}")


if __name__ == "__main__":
    main()
