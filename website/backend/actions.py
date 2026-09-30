"""Long-running actions started from the website: runtime sweeps and building
a dataset sample.

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
    timeout_s: int = 3600              # per pipeline
    retry_failed: bool = False
    force: bool = False

    @classmethod
    def parse(cls, raw: dict) -> "SweepParams":
        try:
            p = cls(source=str(raw["source"]), data=str(raw.get("data") or "input"),
                    timeout_s=int(raw.get("timeout_s") or 3600),
                    retry_failed=bool(raw.get("retry_failed")), force=bool(raw.get("force")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionError(f"bad parameters: {exc}") from None
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
    building = {j["params"].get("name") for j in running("build-sample")
                if j.get("dataset") == ds.name}
    for name in ds.samples:
        info = ds.sample_info(name)
        ok = name not in building
        out.append({"name": name, "label": f"{name}/input/", "ok": ok,
                    "note": "being rebuilt" if not ok else _sample_note(info)})
    return out


def _sample_note(info: dict) -> str | None:
    """Row counts from a sample's manifest: the largest tables first."""
    if not info.get("manifest"):
        return "no sample_manifest.json: how it was built is not recorded"
    rows = sorted(((f, out, src) for f, (out, src) in info["rows"].items() if out is not None),
                  key=lambda r: -(r[2] or r[1]))
    parts = [f"{f} {out:,}" + (f" of {src:,}" if src and src != out else "") + " rows"
             for f, out, src in rows[:2]]
    return "; ".join(parts) or None


def _sweep_source(run: Run, name: str) -> Source:
    src = next((s for s in run.sources if s.name == name), None)
    if src is None or not src.skrub:
        raise ActionError(f"{run.id} has no skrub source {name!r}")
    if not src.files:
        raise ActionError(f"source {name} holds no pipeline files")
    return src


def _sweep_store(run: Run, p: SweepParams) -> tuple[Path, RuntimeStore | None]:
    """The store a sweep writes: the one already holding this source on this
    data (the tool then re-measures only what is missing or stale), else a new
    ``runtime_stats_<source>_<sample folder | fulldata>.json``."""
    for rs in run.runtime:
        if rs.source == p.source and rs.data == p.data and not rs.legacy_rows:
            return rs.path, rs
    tag = "fulldata" if p.data == "input" else p.data
    return run.path / f"runtime_stats_{p.source}_{tag}.json", None


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
# build sample: tools/dataset_sample from the dataset.toml recipe
# --------------------------------------------------------------------------- #
def sample_options(run: Run) -> dict:
    """Whether this run's dataset can build a sample, and why not."""
    ds = run.dataset
    reason = (None if ds.sample_recipe and ds.has_input
              else "no [sample] recipe in dataset.toml" if not ds.sample_recipe
              else "no input/ on this machine to sample from")
    recipe = ds.sample_recipe or {}
    return {"ok": reason is None, "reason": reason, "dataset": ds.name,
            "target": recipe.get("target"), "script": recipe.get("script"),
            "samples": [ds.sample_info(n) for n in ds.samples]}


def _sample_params(raw: dict) -> tuple[int, bool]:
    from dataset_sample.core import SampleError, parse_size
    try:
        size = parse_size(raw.get("size") or "")
    except SampleError as exc:
        raise ActionError(str(exc)) from None
    if size > 10**9:
        raise ActionError("that is not a sample")
    return size, bool(raw.get("force"))


def plan_sample(run: Run, raw: dict) -> dict:
    from dataset_sample.core import SampleError, plan
    opts = sample_options(run)
    if not opts["ok"]:
        raise ActionError(opts["reason"])
    size, force = _sample_params(raw)
    try:
        out = plan(run.dataset.path, size)
    except SampleError as exc:
        raise ActionError(str(exc)) from None
    argv = [sys.executable, "-m", "dataset_sample", str(run.dataset.path), "--size", str(size)]
    if force:
        argv.append("--force")
    return {**out, "force": force, "argv": argv,
            "command": _display(argv, REPO_ROOT)}


def start_sample(run: Run, raw: dict) -> dict:
    """Build a sample. One build per dataset at a time, and never over a sample
    a sweep is reading (the swap would change its data mid-sweep)."""
    p = plan_sample(run, raw)
    if p["exists"] and not p["force"]:
        raise ActionError(f"{p['name']}/ exists; tick rebuild to replace it")
    ds = run.dataset.name
    if any(j.get("dataset") == ds for j in running("build-sample")):
        raise ActionError(f"a sample of {ds} is already being built")
    if p["exists"] and any(j.get("dataset") == ds and j["params"].get("data") == p["name"]
                           for j in running("runtime-sweep")):
        raise ActionError(f"a runtime sweep is reading {p['name']}; rebuild it afterwards")
    return _start({
        "action": "build-sample", "run": run.id, "dataset": ds,
        "label": f"Build sample · {ds}/{p['name']} · {p['size']:,} rows",
        "params": {"name": p["name"], "size": p["size"], "force": p["force"]},
        "argv": p["argv"], "cwd": str(REPO_ROOT), "command": p["command"],
        "outputs": {"sample": p["name"]},
    })


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
        "dataset": spec.get("dataset"),
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
        # a sample belongs to the dataset, so its builds show on every run of it
        if (run is None or spec.get("run") == run.id
                or (spec.get("action") == "build-sample"
                    and spec.get("dataset") == run.dataset.name)):
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
    if any(j.get("dataset") == run.dataset.name and j["params"].get("name") == p["data"]
           for j in running("build-sample")):
        raise ActionError(f"{p['data']} is being rebuilt; wait for the build to finish")
    return _start({
        "action": "runtime-sweep", "run": run.id, "dataset": run.dataset.name,
        "label": f"Runtime sweep · {p['source']} · "
                 f"{p['data'] if p['data'] != 'input' else 'full data'}",
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
