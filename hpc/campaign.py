#!/usr/bin/env python3
"""Plan, execute, and publish independent stratum benchmark cells.

Planning/publishing use Python >=3.11's standard library; executing requires the
project environment. Paths in the manifest are relative to the bound checkout.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
SLURM_KEYS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
              "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_PARTITION",
              "SLURM_JOB_NODELIST")
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "SKRUB_RUST_THREADS",
               "LOKY_MAX_CPU_COUNT")


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    else:
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def relative(path, root=ROOT):
    # Resolve the parent, but do not dereference data symlinks during planning.
    return str(Path(path).absolute().relative_to(root.resolve()))


def inside(root, value):
    path = root / value
    if Path(value).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the checkout: {value}")
    return path


def positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def code_identity(root, run_paths):
    files = {root / "pyproject.toml", root / "uv.lock"}
    for folder in (root / "hpc", root / "tools" / "pipeline_analyzer"):
        files.update(p for p in folder.iterdir()
                     if p.is_file() and p.suffix in {".py", ".sh", ".sbatch", ".def"})
    files.add(root / "website" / "backend" / "registry.py")
    for run in run_paths:
        files.update(run.rglob("*.py"))
        files.update(run.glob("*.toml"))
        files.add(run.parent / "dataset.toml")
    return {relative(p, root): digest(p) for p in sorted(files) if p.is_file()}


def image_identity(image, allow_missing=False):
    image = Path(image).resolve()
    if not image.is_file():
        if allow_missing:
            return None
        raise ValueError(f"missing image: {image}; run bash hpc/build.sh")
    info = image.stat()
    return {"sha256": digest(image), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def plan(config, output, image, *, root=ROOT, allow_missing=False):
    from website.backend.registry import load_dataset

    config = Path(config)
    spec = tomllib.loads(config.read_text())
    allowed = {"name", "repetitions", "modes", "timeout_s", "diagnostics",
               "implementation_selector", "workloads"}
    if set(spec) - allowed:
        raise ValueError(f"unknown campaign keys: {sorted(set(spec) - allowed)}")
    name = spec.get("name", "benchmark")
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("name must contain only letters, digits, '_' or '-'")
    repetitions = positive(spec.get("repetitions", 1), "repetitions")
    timeout = positive(spec.get("timeout_s", 3600), "timeout_s")
    modes = spec.get("modes", ["profile"])
    if (not isinstance(modes, list) or not modes or len(set(modes)) != len(modes)
            or any(m not in ("profile", "wall") for m in modes)):
        raise ValueError("modes must be a nonempty unique list of profile and/or wall")
    selector = spec.get("implementation_selector", "default")
    if selector not in ("default", "greedy"):
        raise ValueError("implementation_selector must be default or greedy")
    if type(spec.get("diagnostics", True)) is not bool:
        raise ValueError("diagnostics must be true or false")
    workloads = spec.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("add at least one [[workloads]] table")

    cells, run_paths = [], set()
    for workload in workloads:
        allowed = {"run", "source", "data", "data_access", "data_version", "only",
                   "scoring", "stage_data"}
        if set(workload) - allowed:
            raise ValueError(f"unknown workload keys: {sorted(set(workload) - allowed)}")
        run_id = workload.get("run", "")
        if not isinstance(run_id, str) or len(Path(run_id).parts) != 2:
            raise ValueError("workload.run must be dataset/run")
        run_path = inside(root, run_id)
        ds = load_dataset(run_path.parent)
        run = next((r for r in ds.runs if r.path == run_path), None)
        if run is None:
            raise ValueError(f"unknown registered run: {run_id}")
        source_name = workload.get("source")
        source = (next((s for s in run.sources if s.name == source_name), None)
                  if source_name else run.default_source())
        if source is None or not source.skrub:
            raise ValueError(f"no DataOps source {source_name!r} in {run_id}")
        data = workload.get("data", "input")
        if not isinstance(data, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", data):
            raise ValueError("data must be input or a sample directory name")
        access = workload.get("data_access", "local")
        if access not in ("local", "remote"):
            raise ValueError("data_access must be local or remote")
        stage = workload.get("stage_data", False)
        if type(stage) is not bool or (stage and access != "local"):
            raise ValueError("stage_data requires local data_access")
        version = workload.get("data_version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("each workload needs a nonempty data_version")
        if access == "remote" and data != "input":
            raise ValueError("remote inputs cannot be selected via a sample cwd; use data=input")
        scoring = workload.get("scoring")
        if scoring is not None and (not isinstance(scoring, str) or not scoring):
            raise ValueError("scoring must be a nonempty scorer string")
        names = list(dict.fromkeys(n for n in run.modules if n in source.files))
        only = workload.get("only")
        if only is not None:
            if not isinstance(only, list) or not only or any(not isinstance(n, str) for n in only):
                raise ValueError("only must be a nonempty list of pipeline names")
            if len(set(only)) != len(only):
                raise ValueError("duplicate pipelines in only")
            unknown = set(only) - set(source.files)
            if run.modules:
                unknown |= set(only) - set(run.modules)
            if unknown:
                raise ValueError(f"unregistered/missing pipelines: {sorted(unknown)}")
            names = only
        if not names:
            raise ValueError(f"no registered pipelines in {run_id}/{source.name}; provide only explicitly")
        run_paths.add(run.path)
        for repetition in range(1, repetitions + 1):
            for mode in modes:
                for pipeline in names:
                    cells.append({
                        "id": len(cells), "run": run.id, "source": source.name,
                        "pipeline": pipeline, "pipeline_path": relative(source.files[pipeline], root),
                        "pipeline_dirs": [relative(d, root) for d in source.dirs],
                        "run_in": relative(ds.path if data == "input" else ds.path / data, root),
                        "data": data, "data_access": access, "data_version": version,
                        "stage_data": stage, "scoring": scoring, "repetition": repetition,
                        "mode": mode, "timeout_s": timeout, "implementation_selector": selector,
                        "diagnostics": mode == "profile" and spec.get("diagnostics", True),
                    })
    output = Path(output).resolve()
    if not output.parent.is_relative_to(root.resolve()):
        raise ValueError("campaign output must be inside the checkout")
    document = {"schema_version": 1, "campaign": name, "created_at": stamp(),
                "config_sha256": digest(config), "config": spec,
                "image": image_identity(image, allow_missing),
                "code_sha256": code_identity(root, run_paths), "cells": cells}
    write_json(output, document, exclusive=True)
    (output.parent / "task_count").write_text(f"{len(cells)}\n")
    shutil.copyfile(config, output.parent / "campaign.toml")
    print(f"{len(cells)} cells; manifest: {output}")
    if document["image"] is None:
        print("Image missing: draft plan only; build the image and re-plan on the cluster.")
    for cell in cells[:12]:
        print(f"  {cell['id']}: {cell['run']} / {cell['source']} / {cell['pipeline']} "
              f"{cell['data']} {cell['mode']} r{cell['repetition']}")
    if len(cells) > 12:
        print(f"  ... {len(cells) - 12} more")
    return document


def validate_code(document, root):
    changed = [name for name, sha in document["code_sha256"].items()
               if not inside(root, name).is_file() or digest(root / name) != sha]
    if changed:
        raise ValueError(f"code changed since planning; create a new campaign: {changed[:5]}")


def data_inventory(folder):
    rows = []
    for path in sorted((folder / "input").rglob("*")):
        if path.is_symlink() and not path.exists():
            raise ValueError(f"dangling input symlink: {path}")
        if path.is_file():
            info = path.stat()
            rows.append({"path": str(path.relative_to(folder)), "bytes": info.st_size,
                         "mtime_ns": info.st_mtime_ns})
    if not rows:
        raise ValueError(f"no local data files under {folder / 'input'}")
    return rows


def environment_info():
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    return {"hostname": platform.node(), "platform": platform.platform(), "cpu": cpu,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "python": sys.version, "slurm": {k: os.environ[k] for k in SLURM_KEYS if k in os.environ},
            "threads": {k: os.environ[k] for k in THREAD_KEYS if k in os.environ},
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}


def pool_fraction():
    """Budget 70% of the smaller of physical RAM and the Slurm allocation."""
    total = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemTotal:"))
    limits = [total]
    if os.environ.get("SLURM_MEM_PER_NODE"):
        limits.append(int(os.environ["SLURM_MEM_PER_NODE"]) * 1024 * 1024)
    # A cgroup limit can further restrict a shared node's job.
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(name).read_text().strip()
            if value != "max":
                limits.append(int(value))
        except (OSError, ValueError):
            pass
    return 0.7 * min(limits) / total


def run_cell(manifest, index, *, root=ROOT, dependency_root=Path("/opt/benchmark")):
    manifest = Path(manifest).resolve()
    document = json.loads(manifest.read_text())
    if index < 0 or index >= len(document["cells"]):
        raise ValueError("cell index out of range")
    cell = document["cells"][index]
    validate_code(document, root)
    if document["image"] is None and not document.get("synthetic"):
        raise ValueError("draft manifest has no image identity; re-plan with the built image")
    if document["image"] is not None:
        current = Path(os.environ["HPC_IMAGE"]).stat()
        expected = document["image"]
        if current.st_size != expected["size"] or current.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError("container image changed since planning; create a new campaign")
    for name in ("pyproject.toml", "uv.lock"):
        if digest(dependency_root / name) != document["code_sha256"][name]:
            raise ValueError(f"container {name} differs from campaign; rebuild the image")
    cell_dir = manifest.parent / "cells" / f"{index:05d}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    with (cell_dir / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f"cell {index} is already running") from None
        if (cell_dir / "completed.json").exists():
            print(f"Cell {index} already completed; retaining its measurement.")
            return 0
        run_in = inside(root, cell["run_in"])
        if not run_in.is_dir():
            raise ValueError(f"missing run directory: {run_in}; stage the data first")
        inputs = data_inventory(run_in) if cell["data_access"] == "local" else []
        sample = run_in / "sample_manifest.json"
        if cell["data"] != "input" and not sample.is_file():
            raise ValueError("sample requires sample_manifest.json beside input/")
        record = {"campaign": document["campaign"], "manifest_sha256": digest(manifest),
                  "cell": cell, "started_at": stamp(), "image": document["image"],
                  "environment": environment_info(), "input_inventory": inputs,
                  "sample_manifest_sha256": digest(sample) if sample.is_file() else None,
                  "data_identity_note": "Inventory uses paths/sizes/mtimes, not content hashes; data_version is caller-supplied."}
        # Isolate writable caches/tmp per cell. Cleanup happens after results are durable.
        with tempfile.TemporaryDirectory(prefix=f"mle-bench-{index}-") as scratch:
            scratch = Path(scratch)
            # Relative model artifacts (e.g. catboost_info/) must not collide
            # when cells use the same dataset. Input paths retain ./input/.
            effective_run_in = scratch / run_in.name
            effective_run_in.mkdir()
            if cell["stage_data"]:
                shutil.copytree(run_in / "input", effective_run_in / "input", symlinks=False)
            elif (run_in / "input").is_dir():
                (effective_run_in / "input").symlink_to(run_in / "input", target_is_directory=True)
            if sample.is_file():
                shutil.copyfile(sample, effective_run_in / sample.name)
            cmd = [sys.executable, "-m", "pipeline_analyzer.runtime"]
            for folder in cell["pipeline_dirs"]:
                cmd += ["--pipelines", str(inside(root, folder))]
            cmd += ["--run-in", str(effective_run_in), "--only", cell["pipeline"],
                    "--out", str(cell_dir / "runtime.json"), "--timeout", str(cell["timeout_s"]),
                    "--force", "--implementation-selector", cell["implementation_selector"],
                    "--buffer-pool-memory-fraction", str(pool_fraction())]
            if cell["scoring"]:
                cmd += ["--scoring", cell["scoring"]]
            if cell["mode"] == "wall":
                cmd.append("--no-stats")
            if cell["diagnostics"]:
                cmd += ["--diagnostics-dir", str(cell_dir / "diagnostics")]
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(root / "tools")]),
                       TMPDIR=str(scratch), MPLCONFIGDIR=str(scratch / "matplotlib"),
                       XDG_CACHE_HOME=str(scratch / "cache"),
                       SKB_DATA_DIRECTORY=str(scratch / "skrub_data"))
            record.update(command=cmd, effective_run_in=str(effective_run_in))
            attempt = len(list(cell_dir.glob("attempt-*.json"))) + 1
            attempt_path = cell_dir / f"attempt-{attempt:03d}.json"
            write_json(attempt_path, record, exclusive=True)
            t0 = time.monotonic()
            with (cell_dir / "runtime.log").open("a") as log:
                log.write(f"\nAttempt {attempt} at {record['started_at']}\n")
                log.flush()
                result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            store_path = cell_dir / "runtime.json"
            store = json.loads(store_path.read_text()) if store_path.is_file() else {}
            entries = store.get("pipelines", {})
            success = result.returncode == 0 and entries.get(cell["pipeline"], {}).get("status") == "ok"
            record.update(finished_at=stamp(), elapsed_s=time.monotonic() - t0,
                          returncode=result.returncode, status="ok" if success else "failed")
            write_json(attempt_path, record)
            if store:
                store.setdefault("meta", {})["hpc"] = record
                write_json(store_path, store)
            if success:
                write_json(cell_dir / "completed.json", record, exclusive=True)
            print(f"Cell {index}: {record['status']}; log: {cell_dir / 'runtime.log'}")
            return 0 if success else 1


def publish(manifest, *, root=ROOT):
    """Copy completed per-cell stores next to runs; never pool repetitions."""
    manifest = Path(manifest).resolve()
    document = json.loads(manifest.read_text())
    if document.get("synthetic"):
        raise ValueError("synthetic smoke artifacts are not corpus runtime stores")
    campaign_id = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    count = 0
    for cell in document["cells"]:
        directory = manifest.parent / "cells" / f"{cell['id']:05d}"
        if not (directory / "completed.json").is_file():
            continue
        store = json.loads((directory / "runtime.json").read_text())
        store["meta"]["measured_run_in"] = store["meta"].get("run_in")
        store["meta"]["run_in"] = str(inside(root, cell["run_in"]))
        store["meta"]["pipelines"] = [str(inside(root, p)) for p in cell["pipeline_dirs"]]
        destination = inside(root, cell["run"]) / f"runtime_stats_hpc_{campaign_id}_{cell['id']:05d}.json"
        encoded = json.dumps(store, indent=2) + "\n"
        if destination.exists():
            if destination.read_text() != encoded:
                raise ValueError(f"refusing to replace a different published result: {destination}")
        else:
            with destination.open("x") as stream:
                stream.write(encoded)
        print(destination)
        count += 1
    if not count:
        raise ValueError("no completed cells to publish")
    print(f"Published {count} independent stores; repetitions remain separate.")


def smoke(*, root=ROOT, dependency_root=Path("/opt/benchmark")):
    """Tiny generated pipeline: test the real worker without corpus data or network."""
    directory = Path(tempfile.mkdtemp(prefix="smoke-", dir=root / "hpc" / "_run"))
    (directory / "input").mkdir()
    (directory / "input" / "train.csv").write_text("x,y\n" + "".join(f"{i},{3*i+2}\n" for i in range(60)))
    (directory / "smoke_features.py").write_text('''from sklearn.linear_model import LinearRegression
def make_model():
    return LinearRegression()
''')
    (directory / "smoke_plan.py").write_text('''import pandas as pd
import skrub
from smoke_features import make_model
from sklearn.model_selection import KFold
data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)
y = data["y"].skb.mark_as_y()
X = data.drop(columns="y").skb.mark_as_X(cv=KFold(3, shuffle=True, random_state=42), split_kwargs={})
pred = X.skb.apply(make_model(), y=y)
''')
    cell = {"id": 0, "run": relative(directory, root), "source": "smoke", "pipeline": "smoke_plan",
            "pipeline_path": relative(directory / "smoke_plan.py", root),
            "pipeline_dirs": [relative(directory, root)], "run_in": relative(directory, root),
            "data": "input", "data_access": "local", "data_version": "synthetic-linear-v1",
            "stage_data": False, "scoring": "r2", "repetition": 1, "mode": "profile",
            "timeout_s": 60, "implementation_selector": "default", "diagnostics": True}
    hashes = code_identity(root, [])
    hashes.update({relative(p, root): digest(p) for p in directory.glob("*.py")})
    manifest = directory / "manifest.json"
    cells = [cell, {**cell, "id": 1, "mode": "wall", "diagnostics": False}]
    write_json(manifest, {"schema_version": 1, "campaign": "smoke", "image": None, "synthetic": True,
                          "code_sha256": hashes, "cells": cells}, exclusive=True)
    for index in range(2):
        result = run_cell(manifest, index, root=root, dependency_root=dependency_root)
        if result:
            raise ValueError(f"smoke failed; inspect {directory / f'cells/{index:05d}/runtime.log'}")
        entry = json.loads((directory / f"cells/{index:05d}/runtime.json").read_text())["pipelines"]["smoke_plan"]
        if entry.get("best_score", 0) < 0.999 or bool(entry.get("ops")) != (index == 0):
            raise ValueError(f"unexpected smoke score or timing mode: {entry.get('best_score')}")
    if not list((directory / "cells/00000/diagnostics").rglob("*.json")):
        raise ValueError("smoke completed without stratum diagnostics")
    print(f"Smoke passed in profile and wall modes: score={entry['best_score']}; artifacts: {directory}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan", help="resolve registered pipelines without importing them")
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--allow-missing-image", action="store_true")
    p = sub.add_parser("run", help="execute one immutable manifest cell inside the image")
    p.add_argument("manifest", type=Path)
    p.add_argument("index", type=int)
    p = sub.add_parser("publish", help="publish completed stores for website discovery")
    p.add_argument("manifest", type=Path)
    sub.add_parser("smoke", help="exercise the real worker on synthetic data inside the image")
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            plan(args.config, args.out, args.image, allow_missing=args.allow_missing_image)
        elif args.action == "run":
            return run_cell(args.manifest, args.index)
        elif args.action == "publish":
            publish(args.manifest)
        else:
            (ROOT / "hpc" / "_run").mkdir(parents=True, exist_ok=True)
            smoke()
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
