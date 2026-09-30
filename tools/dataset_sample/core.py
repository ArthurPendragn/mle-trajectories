"""Build a persisted sample of a dataset from the recipe in its ``dataset.toml``.

A sample is a second run-root beside the full data::

    <dataset>/input/                    full data
    <dataset>/sample_100k/input/        same file names, fewer rows
    <dataset>/sample_100k/sample_manifest.json

so a pipeline runs on it unmodified (``--run-in <dataset>/sample_100k``) and
nothing is patched at run time. The recipe says, per file, how it is reduced::

    [sample]
    target = "Cover_Type"      # stratify sampled tables on this column (optional)
    min_per_class = 5          # keep at least this many rows of every class
    seed = 0

    [sample.tables]
    "train.csv" = "sample"                              # reduced to the requested size
    "test.csv" = "keep"                                 # copied whole (hard link)
    "sample_submission.csv" = { match = "test.csv", key = "id" }   # rows whose key survived

Files the recipe does not list are kept whole. A dataset whose shape needs its
own reasoning (a graph, a data lake) points at a script instead::

    [sample]
    script = "make_input_sample.py"
    out_env = "TTT_OUT"                       # where the script writes input/
    size_env = ["TTT_N_SEED", "TTT_N_DOMAINS"]   # knobs set to the requested size

Sampled CSVs are written by copying the selected records byte for byte, so
formatting, quoting and column order are exactly the source's, and pandas reads
back the same dtypes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ._manifest import MANIFEST, fingerprint

_TABLE_KEYS = {"match", "key", "from_key"}
_SAMPLE_KEYS = {"target", "min_per_class", "seed", "tables", "script", "out_env", "size_env",
                "note"}


class SampleError(Exception):
    """A recipe or dataset the sampler cannot work with; the message says why."""


# --------------------------------------------------------------------------- #
# sizes and names
# --------------------------------------------------------------------------- #
def parse_size(text: str | int) -> int:
    """``100k``, ``1m``, ``100_000``, ``100000`` -> rows."""
    s = str(text).strip().lower().replace("_", "").replace(",", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([km]?)", s)
    if not m:
        raise SampleError(f"not a size: {text!r} (use e.g. 100k, 1m, 250000)")
    n = float(m[1]) * {"": 1, "k": 1_000, "m": 1_000_000}[m[2]]
    if n < 1 or n != int(n):
        raise SampleError(f"not a whole number of rows: {text!r}")
    return int(n)


def size_label(n: int) -> str:
    """100000 -> ``100k``, 1000000 -> ``1m``, 12345 -> ``12345``."""
    for div, suffix in ((1_000_000, "m"), (1_000, "k")):
        if n % div == 0:
            return f"{n // div}{suffix}"
    return str(n)


def sample_name(n: int) -> str:
    return f"sample_{size_label(n)}"


# --------------------------------------------------------------------------- #
# recipe
# --------------------------------------------------------------------------- #
@dataclass
class Rule:
    kind: str                       # sample | keep | match
    match: str | None = None        # match: the file whose surviving keys filter this one
    key: str | None = None          # match: the key column in this file
    from_key: str | None = None     # match: the key column in the other file (default: key)


@dataclass
class Recipe:
    raw: dict
    target: str | None = None
    min_per_class: int = 5
    seed: int = 0
    tables: dict[str, Rule] = field(default_factory=dict)
    script: str | None = None
    out_env: str | None = None
    size_env: list[str] = field(default_factory=list)

    @property
    def sha1(self) -> str:
        return hashlib.sha1(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:16]


def load_recipe(dataset: Path) -> Recipe:
    try:
        cfg = tomllib.loads((dataset / "dataset.toml").read_text())
    except FileNotFoundError:
        raise SampleError(f"{dataset.name}: no dataset.toml") from None
    except tomllib.TOMLDecodeError as exc:
        raise SampleError(f"{dataset.name}/dataset.toml: {exc}") from None
    raw = cfg.get("sample")
    if not isinstance(raw, dict):
        raise SampleError(f"{dataset.name}/dataset.toml has no [sample] recipe")
    unknown = set(raw) - _SAMPLE_KEYS
    if unknown:
        raise SampleError(f"[sample]: unknown key(s) {', '.join(sorted(unknown))}")
    r = Recipe(raw=raw, target=raw.get("target"), min_per_class=int(raw.get("min_per_class", 5)),
               seed=int(raw.get("seed", 0)), script=raw.get("script"), out_env=raw.get("out_env"),
               size_env=[raw["size_env"]] if isinstance(raw.get("size_env"), str)
               else list(raw.get("size_env") or []))
    if r.script:
        if raw.get("tables"):
            raise SampleError("[sample]: a script recipe has no [sample.tables]")
        if not r.out_env or not r.size_env:
            raise SampleError("[sample]: a script recipe needs out_env and size_env")
        return r
    for name, spec in (raw.get("tables") or {}).items():
        if spec in ("sample", "keep"):
            r.tables[name] = Rule(kind=spec)
        elif isinstance(spec, dict) and "match" in spec and "key" in spec \
                and not set(spec) - _TABLE_KEYS:
            r.tables[name] = Rule(kind="match", match=spec["match"], key=spec["key"],
                                  from_key=spec.get("from_key", spec["key"]))
        else:
            raise SampleError(f'[sample.tables] "{name}": expected "sample", "keep" or '
                              '{ match = "<file>", key = "<column>" }')
    if not any(t.kind == "sample" for t in r.tables.values()):
        raise SampleError('[sample.tables]: no table is marked "sample"')
    for name, t in r.tables.items():
        if t.kind == "match" and t.match not in r.tables:
            raise SampleError(f'[sample.tables] "{name}": matches "{t.match}", which is not listed')
    return r


# --------------------------------------------------------------------------- #
# CSV records, byte for byte
# --------------------------------------------------------------------------- #
def _records(path: Path):
    """Yield each CSV record's raw bytes (newline included), header first.

    A record ends at a newline outside quotes, so quoted fields with embedded
    newlines stay whole. Blank lines are skipped, as pandas does."""
    with open(path, "rb") as fh:
        buf, quotes = [], 0
        for line in fh:
            if not buf and not line.strip(b"\r\n"):
                continue
            buf.append(line)
            quotes += line.count(b'"')
            if quotes % 2 == 0:
                yield b"".join(buf)
                buf, quotes = [], 0
        if buf:
            yield b"".join(buf)


def _fields(record: bytes, encoding: str = "utf-8") -> list[str]:
    return next(csv.reader(io.StringIO(record.decode(encoding, errors="replace"))))


def _is_parquet(path: Path) -> bool:
    return path.suffix.lower() in (".parquet", ".pq")


def _is_csv(path: Path) -> tuple[bool, str]:
    suf = path.suffix.lower()
    return suf in (".csv", ".tsv", ".txt"), "\t" if suf == ".tsv" else ","


def _count(path: Path) -> int | None:
    if _is_parquet(path):
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    if _is_csv(path)[0]:
        return sum(1 for _ in _records(path)) - 1
    return None


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(msg, flush=True)


def _link(src: Path, dst: Path) -> None:
    """Hard link, else copy: a kept file costs no space on the same filesystem."""
    src = src.resolve()
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=_link_file)
    else:
        _link_file(src, dst)


def _link_file(src, dst) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _choose(labels: np.ndarray | None, n_rows: int, size: int, recipe: Recipe,
            rng: np.random.Generator) -> np.ndarray:
    """Row positions to keep: a seeded sample of ``size`` rows, stratified
    proportionally on ``labels`` with every class kept at ``min_per_class`` rows
    (or all of them, when it has fewer)."""
    if size >= n_rows:
        return np.arange(n_rows)
    if labels is None:
        return np.sort(rng.choice(n_rows, size, replace=False))
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    want = np.maximum(np.round(counts * size / n_rows).astype(int),
                      np.minimum(counts, recipe.min_per_class))
    keep = []
    for c, k in enumerate(want):
        idx = np.flatnonzero(inverse == c)
        keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def _read_columns(path: Path, cols: list[str], sep: str):
    """Columns of a table as numpy arrays (streamed for CSV)."""
    if _is_parquet(path):
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=cols)
        return {c: t.column(c).to_numpy(zero_copy_only=False) for c in cols}
    import pandas as pd
    parts = {c: [] for c in cols}
    for chunk in pd.read_csv(path, sep=sep, usecols=cols, chunksize=1_000_000, dtype=str,
                             keep_default_na=False):
        for c in cols:
            parts[c].append(chunk[c].to_numpy())
    return {c: np.concatenate(parts[c]) if parts[c] else np.array([], dtype=object) for c in cols}


def _header(path: Path, sep: str) -> list[str]:
    if _is_parquet(path):
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).schema_arrow.names
    first = next(_records(path), b"")
    return next(csv.reader(io.StringIO(first.decode("utf-8", errors="replace")), delimiter=sep))


class _Builder:
    def __init__(self, dataset: Path, recipe: Recipe, size: int, out: Path):
        self.src = dataset / "input"
        self.dataset, self.recipe, self.size, self.out = dataset, recipe, size, out
        self.rng = np.random.default_rng(recipe.seed)
        self.keys: dict[tuple[str, str], set[str]] = {}   # (file, column) -> surviving keys
        self.rows: dict[str, dict] = {}
        self.classes: dict = {}

    def wanted_keys(self, name: str) -> list[str]:
        return sorted({t.from_key for t in self.recipe.tables.values()
                       if t.kind == "match" and t.match == name})

    def run(self) -> None:
        order = self._order()
        for name in order:
            rule = self.recipe.tables.get(name, Rule("keep"))
            path = self.src / name
            if rule.kind == "sample":
                self._sample(name, path)
            elif rule.kind == "match":
                self._match(name, path, rule)
            else:
                self._keep(name, path, listed=name in self.recipe.tables)

    def _order(self) -> list[str]:
        present = sorted(p.name for p in self.src.iterdir())
        for name in self.recipe.tables:
            if name not in present:
                raise SampleError(f"[sample.tables] lists {name!r}, which is not in input/")
        done, order = set(), []

        def visit(name, stack=()):
            if name in done:
                return
            if name in stack:
                raise SampleError(f"[sample.tables]: match cycle through {name!r}")
            rule = self.recipe.tables.get(name)
            if rule and rule.kind == "match":
                visit(rule.match, (*stack, name))
            done.add(name)
            order.append(name)

        for name in present:
            visit(name)
        return order

    # -- rules ---------------------------------------------------------------
    def _keep(self, name: str, path: Path, listed: bool) -> None:
        _link(path, self.out / name)
        n = _count(path) if path.is_file() else None
        self.rows[name] = {"rule": "keep" if listed else "keep (not listed)", "src": n, "out": n}
        for col in self.wanted_keys(name):
            self.keys[(name, col)] = set(map(str, _read_columns(path, [col], _is_csv(path)[1])[col]))
        _log(f"  {name}: kept whole" + (f" ({n:,} rows)" if n is not None else ""))

    def _sample(self, name: str, path: Path) -> None:
        is_csv, sep = _is_csv(path)
        if not (is_csv or _is_parquet(path)):
            raise SampleError(f"{name}: can only sample CSV/TSV or parquet files")
        header = _header(path, sep)
        target = self.recipe.target if self.recipe.target in header else None
        key_cols = self.wanted_keys(name)
        _log(f"  {name}: reading {'target ' + repr(target) if target else 'row count'} ...")
        cols = _read_columns(path, [target], sep) if target else {}
        n_rows = len(cols[target]) if target else _count(path)
        pick = _choose(cols.get(target), n_rows, self.size, self.recipe, self.rng)
        if target:
            src_classes = dict(zip(*[x.tolist() for x in np.unique(cols[target], return_counts=True)]))
            out_classes = dict(zip(*[x.tolist() for x in np.unique(cols[target][pick], return_counts=True)]))
            self.classes[name] = {"target": target, "src": src_classes, "out": out_classes}
        if is_csv:
            self._write_csv_rows(name, path, pick, key_cols, sep)
        else:
            import pyarrow.parquet as pq
            t = pq.read_table(path).take(pick)
            pq.write_table(t, self.out / name)
            for col in key_cols:
                self.keys[(name, col)] = set(map(str, t.column(col).to_pylist()))
        self.rows[name] = {"rule": "sample", "src": n_rows, "out": len(pick)}
        _log(f"  {name}: {len(pick):,} of {n_rows:,} rows"
             + (f", stratified on {target}" if target else ""))

    def _write_csv_rows(self, name, path, pick, key_cols, sep) -> None:
        keep = np.zeros(max(len(pick) and pick[-1] + 1, 1), dtype=bool)
        keep[pick] = True
        header = _header(path, sep)
        idx = {c: header.index(c) for c in key_cols}
        for c in key_cols:
            self.keys[(name, c)] = set()
        records = _records(path)
        n = 0
        with open(self.out / name, "wb") as out:
            out.write(next(records))
            for i, rec in enumerate(records):
                if i < len(keep) and keep[i]:
                    out.write(rec)
                    n += 1
                    if idx:
                        f = next(csv.reader(io.StringIO(rec.decode("utf-8", errors="replace")),
                                            delimiter=sep))
                        for c, j in idx.items():
                            self.keys[(name, c)].add(f[j])
        if n != len(pick):
            raise SampleError(f"{name}: wrote {n} rows, expected {len(pick)} -- the file's "
                              "records do not line up with pandas' rows (unbalanced quotes?)")

    def _match(self, name: str, path: Path, rule: Rule) -> None:
        wanted = self.keys.get((rule.match, rule.from_key))
        if wanted is None:
            raise SampleError(f"{name}: {rule.match} has no column {rule.from_key!r}")
        is_csv, sep = _is_csv(path)
        if _is_parquet(path):
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
            t = pq.read_table(path)
            mask = pc.is_in(pc.cast(t.column(rule.key), pa.string()), value_set=pa.array(sorted(wanted)))
            sub = t.filter(mask)
            pq.write_table(sub, self.out / name)
            n_src, n_out = t.num_rows, sub.num_rows
            for col in self.wanted_keys(name):
                self.keys[(name, col)] = set(map(str, sub.column(col).to_pylist()))
        elif is_csv:
            header = _header(path, sep)
            if rule.key not in header:
                raise SampleError(f"{name}: no column {rule.key!r}")
            j = header.index(rule.key)
            more = {c: header.index(c) for c in self.wanted_keys(name)}
            for c in more:
                self.keys[(name, c)] = set()
            records = _records(path)
            n_src = n_out = 0
            with open(self.out / name, "wb") as out:
                out.write(next(records))
                for rec in records:
                    n_src += 1
                    f = next(csv.reader(io.StringIO(rec.decode("utf-8", errors="replace")),
                                        delimiter=sep))
                    if f[j] in wanted:
                        out.write(rec)
                        n_out += 1
                        for c, k in more.items():
                            self.keys[(name, c)].add(f[k])
        else:
            raise SampleError(f"{name}: can only match CSV/TSV or parquet files")
        self.rows[name] = {"rule": f"match {rule.match}.{rule.from_key} on {rule.key}",
                           "src": n_src, "out": n_out}
        _log(f"  {name}: {n_out:,} of {n_src:,} rows whose {rule.key} survived in {rule.match}")


# --------------------------------------------------------------------------- #
# checks, fingerprint, manifest
# --------------------------------------------------------------------------- #
def _check(src: Path, out: Path, rows: dict, classes: dict) -> list[str]:
    """The sample's own sanity checks; raises on the first that fails."""
    done = []
    want, got = sorted(p.name for p in src.iterdir()), sorted(p.name for p in out.iterdir())
    if want != got:
        raise SampleError(f"file set differs: input/ has {want}, sample has {got}")
    done.append(f"same {len(want)} file(s) as input/")
    for name in want:
        a, b = src / name, out / name
        if a.is_file() and (_is_parquet(a) or _is_csv(a)[0]):
            sep = _is_csv(a)[1]
            if _is_parquet(a):
                import pyarrow.parquet as pq
                same = pq.ParquetFile(a).schema_arrow.equals(pq.ParquetFile(b).schema_arrow)
            else:
                same = _header(a, sep) == _header(b, sep)
            if not same:
                raise SampleError(f"{name}: columns differ from input/")
    done.append("same columns (and parquet schemas) per table")
    for name, r in rows.items():
        if r["src"] and not r["out"]:
            raise SampleError(f"{name}: sample is empty")
    done.append("no table emptied")
    for name, c in classes.items():
        lost = set(c["src"]) - set(c["out"])
        if lost:
            raise SampleError(f"{name}: classes lost: {sorted(lost)}")
        done.append(f"{name}: all {len(c['src'])} classes of {c['target']} present")
    return done


def build(dataset: Path, size: int, *, name: str | None = None, force: bool = False) -> Path:
    """Build ``<dataset>/<name>/`` (default ``sample_<size>``). The sample is built
    in a hidden folder and moved into place only once every check passed, so a
    failed or interrupted build never leaves a half-written sample."""
    dataset = dataset.resolve()
    recipe = load_recipe(dataset)
    if not (dataset / "input").is_dir():
        raise SampleError(f"{dataset.name}: no input/ to sample from")
    name = name or sample_name(size)
    if not re.fullmatch(r"sample[A-Za-z0-9_\-]*", name):
        raise SampleError(f"sample folder names start with 'sample': {name!r}")
    final = dataset / name
    if final.exists() and not force:
        raise SampleError(f"{name}/ exists; rebuild with --force")
    work = dataset / f".{name}.building"
    if work.exists():
        shutil.rmtree(work)
    (work / "input").mkdir(parents=True)
    t0 = time.time()
    _log(f"building {dataset.name}/{name} ({size:,} rows, recipe {recipe.sha1}) ...")
    try:
        if recipe.script:
            rows, classes, extra = _run_script(dataset, recipe, size, work)
        else:
            b = _Builder(dataset, recipe, size, work / "input")
            b.run()
            rows, classes, extra = b.rows, b.classes, {}
        checks = _check(dataset / "input", work / "input", rows, classes)
        for c in checks:
            _log(f"  ✓ {c}")
        manifest = {
            **extra,
            "dataset": dataset.name, "name": name, "size": size,
            "built": datetime.now().astimezone().isoformat(timespec="seconds"),
            "built_by": "tools/dataset_sample",
            "recipe": recipe.raw, "recipe_sha1": recipe.sha1, "seed": recipe.seed,
            "source": "input", "rows": rows, "class_counts": classes, "checks": checks,
            "elapsed_s": round(time.time() - t0, 1),
            "fingerprint": fingerprint(work),
        }
        (work / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    if final.exists():
        old = dataset / f".{name}.old"
        shutil.rmtree(old, ignore_errors=True)
        final.rename(old)
        work.rename(final)
        shutil.rmtree(old, ignore_errors=True)
    else:
        work.rename(final)
    _log(f"done -> {final} in {time.time() - t0:.0f}s (fingerprint {manifest['fingerprint']})")
    return final


def _run_script(dataset: Path, recipe: Recipe, size: int, work: Path):
    """A dataset's own make_sample script, pointed at the hidden build folder."""
    script = (dataset / recipe.script).resolve()
    if dataset not in script.parents or not script.is_file():
        raise SampleError(f"[sample] script {recipe.script!r} is not a file in {dataset.name}/")
    env = {**os.environ, recipe.out_env: str(work / "input"), "PYTHONUNBUFFERED": "1",
           **{k: str(size) for k in recipe.size_env}}
    _log(f"  running {recipe.script} with {', '.join(f'{k}={size}' for k in recipe.size_env)}")
    res = subprocess.run([sys.executable, str(script)], cwd=dataset, env=env)
    if res.returncode:
        raise SampleError(f"{recipe.script} exited with {res.returncode}")
    extra = {}
    own = work / MANIFEST       # scripts that write their own manifest keep theirs
    if own.is_file():
        extra = {"script_manifest": json.loads(own.read_text())}
    rows = {}
    for p in sorted((work / "input").iterdir()):
        src = dataset / "input" / p.name
        rows[p.name] = {"rule": f"script {recipe.script}", "out": _count(p) if p.is_file() else None,
                        "src": None}
        if src.is_file() and _is_parquet(src):
            rows[p.name]["src"] = _count(src)
    extra["script"] = recipe.script
    extra["knobs"] = {k: size for k in recipe.size_env}
    return rows, {}, extra


def adopt(folder: Path, note: str) -> Path:
    """Write a manifest for a sample built before this tool existed: what it holds
    (row counts, fingerprint), with ``note`` saying how it was made. The recipe
    and knobs are not known, so they are not claimed."""
    folder = folder.resolve()
    if not (folder / "input").is_dir():
        raise SampleError(f"{folder} has no input/")
    if (folder / MANIFEST).exists():
        raise SampleError(f"{folder.name} already has a {MANIFEST}")
    src = folder.parent / "input"
    rows = {}
    for p in sorted((folder / "input").iterdir()):
        s = src / p.name
        rows[p.name] = {"out": _count(p) if p.is_file() else None,
                        "src": _count(s) if s.is_file() and _is_parquet(s) else None}
        _log(f"  {p.name}: {rows[p.name]['out']}")
    manifest = {"dataset": folder.parent.name, "name": folder.name, "adopted": True,
                "built_by": "unknown (manifest written afterwards by tools/dataset_sample)",
                "note": note, "rows": rows, "fingerprint": fingerprint(folder),
                "written": datetime.now().astimezone().isoformat(timespec="seconds")}
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return folder / MANIFEST


def plan(dataset: Path, size: int, name: str | None = None) -> dict:
    """What ``build`` would do, without reading any data."""
    dataset = dataset.resolve()
    recipe = load_recipe(dataset)
    name = name or sample_name(size)
    src = dataset / "input"
    files = sorted(p.name for p in src.iterdir()) if src.is_dir() else []
    if recipe.script:
        tables = {f: f"script {recipe.script}" for f in files}
    else:
        tables = {}
        for f in files:
            r = recipe.tables.get(f)
            tables[f] = ("keep (not listed)" if r is None else r.kind if r.kind != "match"
                         else f"rows whose {r.key} survived in {r.match}")
    return {"dataset": dataset.name, "name": name, "size": size, "exists": (dataset / name).exists(),
            "has_input": src.is_dir(), "recipe_sha1": recipe.sha1, "target": recipe.target,
            "script": recipe.script, "tables": tables}
