"""Background analysis jobs and their on-disk cache.

A job builds one run's operator analysis for one source (``worker.py``). Its
state lives entirely in files under ``website/.cache/analysis/<dataset>/<run>/``,
so it survives an API restart (``--reload`` restarts on every code change):

    <source>-<key>.json   done
    <source>-<key>.pid    running (while that pid is alive)
    <source>-<key>.log    the worker's output; a log with neither of the above
                          means the job failed

``key`` hashes everything the result depends on -- every ``.py`` file in the
source's folders (shared modules included), the lineage file, the stratum build,
the analyzer's own code and the fold flag -- so any change produces a new key and
a fresh build, and nothing has to be invalidated by hand.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from .registry import REPO_ROOT, Run, Source, current_stratum_commit

CACHE = REPO_ROOT / "website" / ".cache" / "analysis"
MAX_RUNNING = 2          # a data-lake run alone takes ~1 GB while it builds
_ANALYZER = REPO_ROOT / "tools" / "pipeline_analyzer"


def _file_digest(h, path: Path) -> None:
    h.update(str(path).encode())
    h.update(hashlib.sha1(path.read_bytes()).digest())


def cache_key(run: Run, source: Source) -> str:
    h = hashlib.sha1()
    for d in source.dirs:
        for f in sorted(d.glob("*.py")):
            _file_digest(h, f)
    lineage = run.trajectory_file or (source.dirs[0] / "results.json")
    if lineage.is_file():
        _file_digest(h, lineage)
    for f in sorted(_ANALYZER.glob("*.py")) + [Path(__file__).with_name("worker.py")]:
        _file_digest(h, f)
    h.update(f"{current_stratum_commit()}|{source.fold_identical_code}".encode())
    return h.hexdigest()[:16]


def _paths(run: Run, source: Source) -> dict[str, Path]:
    base = CACHE / run.dataset.name / run.name / f"{source.name}-{cache_key(run, source)}"
    return {k: base.with_suffix(f".{k}") for k in ("json", "pid", "log")}


def _alive(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def _running() -> int:
    return sum(1 for p in CACHE.glob("*/*/*.pid") if _alive(p))


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def status(run: Run, source: Source) -> dict:
    """``ready`` (with ``path``), ``running``, ``failed`` (with the log tail) or
    ``missing``."""
    p = _paths(run, source)
    if p["json"].is_file():
        if p["pid"].is_file() and not _alive(p["pid"]):
            p["pid"].unlink(missing_ok=True)
        return {"status": "ready", "path": p["json"]}
    if p["pid"].is_file():
        if _alive(p["pid"]):
            return {"status": "running", "log": _tail(p["log"], 5)}
        p["pid"].unlink(missing_ok=True)
        if p["json"].is_file():
            return {"status": "ready", "path": p["json"]}
    if p["log"].is_file():
        return {"status": "failed", "log": _tail(p["log"])}
    return {"status": "missing"}


def start(run: Run, source: Source, *, retry: bool = False) -> dict:
    """Start the job unless it is done, running, or (without ``retry``) failed.
    Returns the resulting status; ``queued`` when too many jobs are running."""
    st = status(run, source)
    if st["status"] in ("ready", "running") or (st["status"] == "failed" and not retry):
        return st
    if _running() >= MAX_RUNNING:
        return {"status": "queued"}
    p = _paths(run, source)
    p["json"].parent.mkdir(parents=True, exist_ok=True)
    # drop results of older keys for this source: they describe files that changed
    for old in p["json"].parent.glob(f"{source.name}-*.*"):
        if old.stem != p["json"].stem and old.suffix in (".json", ".log"):
            old.unlink(missing_ok=True)
    with open(p["log"], "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "website.backend.worker", run.id, source.name, str(p["json"])],
            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,      # outlives an API reload
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    p["pid"].write_text(str(proc.pid))
    return {"status": "running", "log": ""}


def cached_result(run: Run, source: Source) -> Path | None:
    p = _paths(run, source)["json"]
    return p if p.is_file() else None
