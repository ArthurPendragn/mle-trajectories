"""Long-running actions started from the website. So far: runtime sweeps.

A job is a folder ``website/.cache/actions/<id>/``, so its state survives an API
restart (``--reload`` restarts on every code change):

    spec.json   what was asked, the exact command, what it writes
    log         the command's output
    pid         the runner process, while it lives
    exit        {"returncode", "stopped", "finished_at"} once it ended

The runner (``python -m website.backend.actions <job dir>``) starts the command
in its own session, waits, and records the exit; stopping a job signals the
runner, which takes down the whole process tree (the runtime tool starts each
pipeline in a session of its own, so a process-group kill would miss it).

Every command is built here from a registry run and validated parameters; the
frontend sends parameters, never a command or a path.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .registry import REPO_ROOT, Run, RuntimeStore, Source

CACHE = REPO_ROOT / "website" / ".cache" / "actions"
_JOB_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")


class ActionError(ValueError):
    """A request the run cannot support; the message is shown to the user."""


# --------------------------------------------------------------------------- #
# runtime sweep: pipeline_analyzer.runtime over one source, on one data size
# --------------------------------------------------------------------------- #
@dataclass
class SweepParams:
    source: str
    data: str = "input"                # "input" (the full data) or a sample folder
    sample_rows: int | None = None     # --sample-rows: cap every pandas.read_csv
    timeout_s: int = 3600              # per pipeline
    retry_failed: bool = False
    force: bool = False

    @classmethod
    def parse(cls, raw: dict) -> "SweepParams":
        try:
            p = cls(source=str(raw["source"]), data=str(raw.get("data") or "input"),
                    sample_rows=None if raw.get("sample_rows") in (None, "", 0)
                    else int(raw["sample_rows"]),
                    timeout_s=int(raw.get("timeout_s") or 3600),
                    retry_failed=bool(raw.get("retry_failed")), force=bool(raw.get("force")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionError(f"bad parameters: {exc}") from None
        if p.sample_rows is not None and not 1 <= p.sample_rows <= 10**10:
            raise ActionError("row cap must be a positive number")
        if not 10 <= p.timeout_s <= 7 * 86400:
            raise ActionError("timeout must be between 10 s and 7 days")
        return p


def data_options(run: Run) -> list[dict]:
    """The data a sweep can run on: the dataset's ``input/`` and each sample folder."""
    ds = run.dataset
    status = ds.data_status()
    full = {"name": "input", "label": "full data (input/)",
            "ok": status != "missing", "note": None}
    if status == "remote":
        full.update(label="full data", note=f"pipelines read {ds.data}")
    elif status == "missing":
        full["note"] = "no input/ on this machine"
    out = [full]
    for name in ds.samples:
        out.append({"name": name, "label": f"{name}/input/", "ok": True,
                    "note": _sample_note(ds.path / name)})
    return out


def _sample_note(folder: Path) -> str | None:
    """Row counts from a make_sample.py manifest, when the sample has one."""
    try:
        rows = json.loads((folder / "sample_manifest.json").read_text()).get("rows") or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    parts = [f"{f} {r['out']:,} of {r['src']:,} rows" for f, r in rows.items()
             if isinstance(r, dict) and "out" in r and "src" in r]
    return "; ".join(parts[:2]) or None


def _sweep_source(run: Run, name: str) -> Source:
    src = next((s for s in run.sources if s.name == name), None)
    if src is None or not src.skrub:
        raise ActionError(f"{run.id} has no skrub source {name!r}")
    if not src.files:
        raise ActionError(f"source {name} holds no pipeline files")
    return src


def _sweep_store(run: Run, p: SweepParams) -> tuple[Path, RuntimeStore | None]:
    """The store a sweep writes: the one already holding this source at this
    data size (the tool then re-measures only what is missing or stale), else a
    new ``runtime_stats_<source>[_<sample>][_<N>rows|_fulldata].json``."""
    for rs in run.runtime:
        if rs.source == p.source and rs.data == p.data and rs.sample_rows == p.sample_rows:
            return rs.path, rs
    tag = "" if p.data == "input" else f"_{p.data}"
    tag += f"_{p.sample_rows}rows" if p.sample_rows else ("_fulldata" if p.data == "input" else "")
    return run.path / f"runtime_stats_{p.source}{tag}.json", None


def plan_sweep(run: Run, raw: dict, *, listing: bool = False) -> dict:
    """What a sweep with these parameters would do: the store, the command, and
    (``listing``) the tool's own ``--list`` of what it would measure."""
    p = SweepParams.parse(raw)
    src = _sweep_source(run, p.source)
    opt = next((o for o in data_options(run) if o["name"] == p.data), None)
    if opt is None:
        raise ActionError(f"no data folder {p.data!r} for {run.dataset.name}")
    if not opt["ok"]:
        raise ActionError(f"{opt['label']}: {opt['note']}")
    run_in = run.dataset.path if p.data == "input" else run.dataset.path / p.data
    store, existing = _sweep_store(run, p)
    names = [m for m in run.modules if m in src.files] or sorted(src.files)
    argv = [sys.executable, "-m", "pipeline_analyzer.runtime"]
    for d in src.dirs:
        argv += ["--pipelines", str(d)]
    argv += ["--run-in", str(run_in), "--out", str(store), "--timeout", str(p.timeout_s)]
    if p.sample_rows:
        argv += ["--sample-rows", str(p.sample_rows)]
    if p.retry_failed:
        argv.append("--retry-failed")
    if p.force:
        argv.append("--force")
    argv += ["--only", *names]
    plan = {
        "params": p.__dict__, "argv": argv, "cwd": str(run.path),
        "store": store.name, "store_exists": existing is not None,
        "store_summary": None if existing is None else {
            "n_ok": existing.n_ok, "n_failed": existing.n_failed, "n_stale": existing.n_stale},
        "n_pipelines": len(names),
        "command": _display(argv, run.path),
    }
    if listing:
        plan["listing"] = _list(argv, run.path)
    return plan


def _display(argv: list[str], cwd: Path) -> str:
    """The command as one would type it from the run folder (``--only`` elided)."""
    out, skip = ["python"], False
    for a in argv[1:]:
        if a == "--only":
            out.append("--only …")
            skip = True
        elif not skip:
            try:
                a = os.path.relpath(a, cwd) if a.startswith("/") else a
            except ValueError:
                pass
            out.append(a)
    return " ".join(out)


def _list(argv: list[str], cwd: Path) -> dict:
    """Run the tool's ``--list`` (cheap: it only imports stratum for its version)."""
    try:
        res = subprocess.run([*argv, "--list"], cwd=cwd, capture_output=True, text=True,
                             timeout=180, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "--list timed out"}
    text = (res.stdout + res.stderr).strip()
    m = re.search(r"would run (\d+) of (\d+)", text)
    return {"ok": res.returncode == 0, "text": text,
            "would_run": int(m.group(1)) if m else None,
            "total": int(m.group(2)) if m else None}


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _job_dir(job_id: str) -> Path:
    if not _JOB_ID.match(job_id):
        raise ActionError("no such job")
    d = CACHE / job_id
    if not (d / "spec.json").is_file():
        raise ActionError("no such job")
    return d


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _pid(d: Path) -> int | None:
    try:
        return int((d / "pid").read_text())
    except (OSError, ValueError):
        return None


def _state(d: Path) -> str:
    """``running``, ``done``, ``failed``, ``stopped`` or ``lost`` (the runner died
    without recording an exit, e.g. the node rebooted)."""
    ended = _read_json(d / "exit")
    if ended:
        return "stopped" if ended.get("stopped") else "done" if ended.get("returncode") == 0 else "failed"
    pid = _pid(d)
    return "running" if pid and _alive(pid) else "lost" if pid else "starting"


_COUNTS = re.compile(r"^(\d+) pipeline\(s\), (\d+) to measure \((\d+) cached")
_SUMMARY = re.compile(r"^(\d+) measured \((\d+) failed\)")
_STEP = re.compile(r"^\[(\d+)/(\d+)\] (\S+) … ?(.*)$")


def _progress(log: str) -> dict:
    """Parse the runtime tool's own progress lines."""
    out: dict = {"total": None, "todo": None, "cached": None, "done": 0, "failed": 0,
                 "current": None, "finished": False}
    for line in log.splitlines():
        if m := _SUMMARY.match(line):
            out.update(finished=True, failed=int(m[2]), current=None)
        elif m := _COUNTS.match(line):
            out.update(total=int(m[1]), todo=int(m[2]), cached=int(m[3]))
        elif m := _STEP.match(line):
            if m[4]:
                out["done"] = int(m[1])
                out["failed"] += not m[4].startswith("ok")
                out["current"] = None
            else:
                out["current"] = m[3]
    return out


def _tail(path: Path, n: int) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def job_status(job_id: str, *, log_lines: int = 30) -> dict:
    d = _job_dir(job_id)
    spec = _read_json(d / "spec.json") or {}
    ended = _read_json(d / "exit") or {}
    try:
        log = (d / "log").read_text(errors="replace")
    except OSError:
        log = ""
    progress = _progress(log)
    state = _state(d)
    # the runtime tool exits 1 when any pipeline failed; that is a finished sweep
    # with failures (shown in the progress), not a failed job
    if state == "failed" and progress["finished"]:
        state = "done"
    return {
        "id": job_id, "action": spec.get("action"), "run": spec.get("run"),
        "label": spec.get("label"), "params": spec.get("params"),
        "command": spec.get("command"), "outputs": spec.get("outputs", {}),
        "started_at": spec.get("started_at"), "finished_at": ended.get("finished_at"),
        "returncode": ended.get("returncode"), "state": state,
        "progress": progress,
        "log": "\n".join(log.splitlines()[-log_lines:]),
    }


def jobs_for(run: Run | None = None, limit: int = 10) -> list[dict]:
    if not CACHE.is_dir():
        return []
    ids = sorted((p.name for p in CACHE.iterdir() if _JOB_ID.match(p.name)), reverse=True)
    out = []
    for job_id in ids:
        spec = _read_json(CACHE / job_id / "spec.json") or {}
        if run is None or spec.get("run") == run.id:
            out.append(job_status(job_id, log_lines=20))
        if len(out) >= limit:
            break
    return out


def running(action: str) -> list[dict]:
    return [j for j in jobs_for(limit=50) if j["action"] == action and j["state"] in ("running", "starting")]


def start_sweep(run: Run, raw: dict) -> dict:
    """Start a runtime sweep. One at a time on the node: two sweeps side by side
    would each slow the other down, and the timings are the measurement."""
    plan = plan_sweep(run, raw)
    busy = running("runtime-sweep")
    if busy:
        raise ActionError(f"a runtime sweep is already running ({busy[0]['run']}); "
                          "stop it or wait for it to finish")
    p = plan["params"]
    size = (p["data"] if p["data"] != "input" else "full data") + (
        f", {p['sample_rows']:,} rows" if p["sample_rows"] else "")
    return _start({
        "action": "runtime-sweep", "run": run.id,
        "label": f"Runtime sweep · {p['source']} · {size}",
        "params": p, "argv": plan["argv"], "cwd": plan["cwd"], "command": plan["command"],
        "outputs": {"runtime": Path(plan["store"]).stem},
    })


def _start(spec: dict) -> dict:
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    d = CACHE / job_id
    d.mkdir(parents=True)
    spec = {"id": job_id, "started_at": _now(), **spec}
    (d / "spec.json").write_text(json.dumps(spec, indent=1))
    with open(d / "log", "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "website.backend.actions", str(d)],
            cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,      # outlives an API reload
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    (d / "pid").write_text(str(proc.pid))
    return job_status(job_id)


def stop(job_id: str) -> dict:
    d = _job_dir(job_id)
    pid = _pid(d)
    if _state(d) == "running" and pid:
        os.kill(pid, signal.SIGTERM)       # the runner takes down its tree
        for _ in range(40):
            if (d / "exit").is_file() or not _alive(pid):
                break
            time.sleep(0.25)
    return job_status(job_id)


# --------------------------------------------------------------------------- #
# the runner process
# --------------------------------------------------------------------------- #
def _kill_tree(root: int) -> None:
    import psutil
    try:
        procs = psutil.Process(root).children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def _runner(d: Path) -> int:
    spec = json.loads((d / "spec.json").read_text())
    stopped = False

    def on_term(signum, frame):
        nonlocal stopped
        stopped = True
        print(f"\n! stopped from the website at {_now()}", flush=True)
        _kill_tree(os.getpid())

    signal.signal(signal.SIGTERM, on_term)
    print(f"$ {spec['command']}", flush=True)
    proc = subprocess.Popen(spec["argv"], cwd=spec["cwd"], stdin=subprocess.DEVNULL,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
    while True:
        try:
            rc = proc.wait()
            break
        except InterruptedError:
            continue
    (d / "exit").write_text(json.dumps({"returncode": rc, "stopped": stopped,
                                        "finished_at": _now()}))
    return rc


if __name__ == "__main__":
    raise SystemExit(_runner(Path(sys.argv[1])))
