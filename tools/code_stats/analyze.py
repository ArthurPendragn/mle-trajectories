"""Static features of one pipeline script: what can be said from its source alone.

Nothing is imported or run; a file is parsed with ``ast`` and walked once. So
the features are the *written* code, which is what an agent produced -- not what
executed (a branch that never runs still counts, a model built in a loop counts
once). Names are resolved through the file's own imports, so ``lgb.LGBMRegressor``
after ``import lightgbm as lgb`` is ``lightgbm.LGBMRegressor``; a receiver's type
is not known, so method counts are by method *name*, restricted to names that are
specific to pandas or polars.
"""
from __future__ import annotations

import ast
import re
from collections import Counter

from .catalog import classify

# --- what counts as what ----------------------------------------------------- #
READERS = {"read_csv", "read_parquet", "read_table", "read_feather", "read_json", "read_excel",
           "read_pickle", "read_ipc", "scan_csv", "scan_parquet", "scan_ipc", "scan_ndjson",
           "read_ndjson", "read_database", "load", "loadtxt", "open_dataset", "dataset"}
WRITERS = {"to_csv", "to_parquet", "to_feather", "to_json", "to_pickle", "write_csv",
           "write_parquet", "write_ipc", "save", "savez", "savetxt", "dump", "write_table"}
# method names specific enough to attribute to a library without knowing the receiver
PANDAS_METHODS = {"assign", "merge", "groupby", "agg", "aggregate", "pivot_table", "pivot", "melt",
                  "fillna", "dropna", "drop_duplicates", "value_counts", "apply", "map", "replace",
                  "rename", "sort_values", "reset_index", "set_index", "query", "shift", "rolling",
                  "ewm", "cumsum", "rank", "get_dummies", "to_datetime", "to_numeric", "astype",
                  "select_dtypes", "isna", "isnull", "notna", "nunique", "concat", "cut", "qcut",
                  "iterrows", "itertuples", "transform", "explode", "stack", "unstack", "drop", "isin",
                  "merge_asof", "describe", "corr", "interpolate", "clip", "nlargest", "nsmallest"}
POLARS_METHODS = {"with_columns", "with_columns_seq", "group_by", "collect", "lazy", "col", "lit",
                  "when", "with_row_index", "with_row_count", "unnest", "over", "scan_csv",
                  "scan_parquet", "sink_parquet", "sink_csv", "join_asof", "pipe"}
SEED_KWARGS = {"random_state", "seed", "random_seed", "rng"}
SEED_CALLS = {"seed", "manual_seed", "manual_seed_all", "set_seed", "default_rng"}

_LITERAL_MAX = 80


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _value(node: ast.AST, consts: dict[str, object]):
    """A literal value, a string built from literals/known constants, or the
    expression's source (marked with a leading ``=``) when it is not literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass
    s = _string(node, consts)
    if s is not None:
        return s
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    text = ast.unparse(node)
    return "=" + (text if len(text) <= _LITERAL_MAX else text[:_LITERAL_MAX] + "…")


def _string(node: ast.AST, consts: dict[str, object]) -> str | None:
    """A path-like string assembled from literals: f-strings, ``+``, ``/`` on
    Path, ``os.path.join``. Unknown pieces become ``{…}``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        v = consts.get(node.id)
        return v if isinstance(v, str) else None
    if isinstance(node, ast.JoinedStr):
        out = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                out.append(str(v.value))
            else:
                inner = _string(v.value, consts) if isinstance(v, ast.FormattedValue) else None
                out.append(inner if inner is not None else "{…}")
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        a, b = _string(node.left, consts), _string(node.right, consts)
        if a is None and b is None:
            return None
        sep = "/" if isinstance(node.op, ast.Div) else ""
        return f"{a if a is not None else '{…}'}{sep}{b if b is not None else '{…}'}"
    if isinstance(node, ast.Call):
        name = _dotted(node.func) or ""
        if name.endswith(("path.join", "Path")) and node.args:
            parts = [_string(a, consts) for a in node.args]
            if any(p is not None for p in parts):
                return "/".join(p if p is not None else "{…}" for p in parts)
    return None


def _path(p: str | None) -> str | None:
    """``./input/x.csv`` and ``input/x.csv`` are the same file, and so are a
    dataset folder with and without its trailing slash. URLs (``gs://``) keep
    their scheme."""
    if p is None:
        return None
    while p.startswith("./"):
        p = p[2:]
    scheme, sep, rest = p.partition("://")
    if not sep:
        scheme, rest = "", p
    while "//" in rest:
        rest = rest.replace("//", "/")
    if len(rest) > 1:
        rest = rest.rstrip("/")
    return f"{scheme}://{rest}" if sep else rest


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.alias: dict[str, str] = {}          # local name -> qualified module/object
        self.consts: dict[str, object] = {}      # module-level string constants
        self.imports: set[str] = set()
        self.structure = Counter()
        self.depth = self.max_depth = 0
        self.complexity = 1
        self.components: list[dict] = []
        self.calls = Counter()                   # qualified function calls (lowercase names)
        self.pandas, self.polars = Counter(), Counter()
        self.reads: list[dict] = []
        self.writes: list[dict] = []
        self.col_writes = self.loc_writes = self.inplace = 0
        self.columns_written: set[str] = set()
        self.pip: list[str] = []
        self.shell = 0
        self.gpu = False
        self.seeds: set[str] = set()
        self.prints = 0
        self.networks: list[str] = []            # classes subclassing nn.Module
        self.class_stack: list[str] = []
        self.used_inside: set[str] = set()       # names called inside a network's body

    # -- imports and constants ------------------------------------------------
    def visit_Import(self, node):
        for a in node.names:
            self.imports.add(a.name.split(".")[0])
            self.alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and not node.level:
            self.imports.add(node.module.split(".")[0])
            for a in node.names:
                self.alias[a.asname or a.name] = f"{node.module}.{a.name}"
        self.generic_visit(node)

    def qualify(self, dotted: str | None) -> str | None:
        if not dotted:
            return None
        head, _, rest = dotted.partition(".")
        base = self.alias.get(head)
        if base is None:
            return dotted
        return f"{base}.{rest}" if rest else base

    # -- structure ---------------------------------------------------------------
    def _block(self, node, key):
        self.structure[key] += 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.generic_visit(node)
        self.depth -= 1

    def visit_FunctionDef(self, node):
        self._block(node, "functions")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        bases = [self.qualify(_dotted(b)) or "" for b in node.bases]
        if any(b.endswith("nn.Module") or b == "torch.nn.Module" for b in bases):
            self.networks.append(node.name)
        self.class_stack.append(node.name)
        self._block(node, "classes")
        self.class_stack.pop()

    def visit_For(self, node):
        self.complexity += 1
        self._block(node, "for")

    visit_AsyncFor = visit_For

    def visit_While(self, node):
        self.complexity += 1
        self._block(node, "while")

    def visit_If(self, node):
        self.complexity += 1
        self._block(node, "if")

    def visit_Try(self, node):
        self.complexity += len(node.handlers)
        self._block(node, "try")

    visit_TryStar = visit_Try

    def visit_With(self, node):
        self._block(node, "with")

    visit_AsyncWith = visit_With

    def visit_IfExp(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.complexity += 1 + len(node.ifs)
        self.generic_visit(node)

    def _comp(self, node):
        self.structure["comprehensions"] += 1
        self.generic_visit(node)

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    def visit_Lambda(self, node):
        self.structure["lambdas"] += 1
        self.generic_visit(node)

    # -- assignments -------------------------------------------------------------
    def visit_Assign(self, node):
        if self.depth == 0 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            # module-level constants (N_SPLITS = 5, DATA = "./input/train.csv"), so a
            # parameter or path given by name still shows its value; the latest
            # assignment before the use wins, as the walk is in source order
            name = node.targets[0].id
            try:
                self.consts[name] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                s = _string(node.value, self.consts)
                if s is not None:
                    self.consts[name] = s
                else:
                    self.consts.pop(name, None)
        for t in node.targets:
            self._target(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._target(node.target)
        self.generic_visit(node)

    def _target(self, t):
        if isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._target(e)
            return
        if not isinstance(t, ast.Subscript):
            return
        if isinstance(t.value, ast.Attribute) and t.value.attr in ("loc", "iloc", "at", "iat"):
            self.loc_writes += 1
            return
        key = t.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            self.col_writes += 1
            self.columns_written.add(key.value)
        elif isinstance(key, ast.JoinedStr):
            self.col_writes += 1
            self.columns_written.add(_string(key, self.consts) or "{…}")
        elif isinstance(key, (ast.List, ast.Tuple)) and key.elts and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in key.elts):
            self.col_writes += 1
            self.columns_written.update(e.value for e in key.elts)

    # -- calls -------------------------------------------------------------------
    def visit_Call(self, node):
        dotted = _dotted(node.func)
        qual = self.qualify(dotted)
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None)
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}
        spread = [k.value for k in node.keywords if k.arg is None]     # f(**params)

        for k, v in kwargs.items():
            if k == "inplace" and isinstance(v, ast.Constant) and v.value is True:
                self.inplace += 1
            if k in SEED_KWARGS:
                val = _value(v, self.consts)
                self.seeds.add(str(val))
            if isinstance(v, ast.Constant) and isinstance(v.value, str) and (
                    v.value.lower().startswith(("cuda", "gpu")) or v.value in ("GPU", "gpu_hist")):
                self.gpu = True

        if attr == "print" and isinstance(node.func, ast.Name):
            self.prints += 1
        if isinstance(node.func, ast.Name) and any(c in self.networks for c in self.class_stack):
            self.used_inside.add(node.func.id)
        if attr in SEED_CALLS and qual and qual.split(".")[0] in ("numpy", "torch", "random", "tensorflow", "transformers", "lightning"):
            if node.args:
                self.seeds.add(str(_value(node.args[0], self.consts)))
        if qual and (qual.startswith("torch.cuda") or qual.endswith(".cuda")):
            self.gpu = True

        # shell and pip
        if qual and (qual.startswith("subprocess.") or qual in ("os.system", "os.popen")):
            self.shell += 1
            words = []
            for a in node.args[:1]:
                if isinstance(a, (ast.List, ast.Tuple)):
                    words = [e.value for e in a.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                elif isinstance(a, ast.Constant) and isinstance(a.value, str):
                    words = a.value.split()
            if "pip" in words and "install" in words:
                after = words[words.index("install") + 1:]
                self.pip.extend(w for w in after if not w.startswith("-"))

        # data I/O
        if attr in READERS and node.args or attr in READERS and kwargs.get("path") is not None:
            lib = (qual or "").split(".")[0]
            if attr not in ("load", "dataset", "open_dataset") or lib in ("numpy", "pyarrow", "torch", "joblib", "pickle", "json"):
                arg = node.args[0] if node.args else kwargs.get("path")
                self.reads.append({"func": f"{lib}.{attr}" if lib and lib != attr else attr,
                                   "path": _path(_string(arg, self.consts)), "line": node.lineno})
        elif attr in WRITERS:
            arg = node.args[0] if node.args else (kwargs.get("path") or kwargs.get("path_or_buf"))
            path = _path(_string(arg, self.consts)) if arg is not None else None
            self.writes.append({"func": attr, "path": path, "line": node.lineno})

        # pandas / polars specifics
        if attr in PANDAS_METHODS and (isinstance(node.func, ast.Attribute)):
            root = (qual or "").split(".")[0]
            if root not in ("polars", "torch", "numpy", "sklearn", "os", "re", "json"):
                self.pandas[attr] += 1
        if attr in POLARS_METHODS:
            root = (qual or "").split(".")[0]
            if root == "polars" or attr in ("with_columns", "group_by", "collect", "lazy",
                                            "with_row_index", "with_row_count", "sink_parquet"):
                self.polars[attr] += 1

        # components: classes (and functional training APIs) from ML libraries
        if qual:
            info = classify(qual)
            if info:
                self.components.append({
                    **info, "line": node.lineno,
                    "params": {**{k: _value(v, self.consts) for k, v in kwargs.items()},
                               **({"**": " ".join("=" + ast.unparse(s) for s in spread)} if spread else {})},
                    "n_args": len(node.args),
                })
            elif attr and attr[:1].islower() and "." in qual:
                self.calls[qual] += 1
        self.generic_visit(node)


def _docstring_lines(tree: ast.Module) -> int:
    n = 0
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            n += body[0].end_lineno - body[0].lineno + 1
    return n


def analyze_source(src: str) -> dict:
    """Features of one script. A file that does not parse returns ``ok: False``
    with the error and only its line counts."""
    lines = src.splitlines()
    blank = sum(1 for ln in lines if not ln.strip())
    comment = sum(1 for ln in lines if ln.strip().startswith("#"))
    size = {"lines": len(lines), "blank": blank, "comment": comment}
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"ok": False, "error": f"line {exc.lineno}: {exc.msg}",
                "size": {**size, "loc": len(lines) - blank - comment}}
    doc = _docstring_lines(tree)
    size.update(docstring=doc, loc=len(lines) - blank - comment - doc)
    v = _Visitor()
    v.visit(tree)

    # a network the file defines is a model, unless it is only a building block
    # of another network (a ResidualBlock used inside the model class)
    for name in v.networks:
        kind = ("loss" if name.endswith("Loss") else "layer" if name in v.used_inside
                else "model")
        v.components.append({"name": name, "qualified": name, "lib": "torch", "kind": kind,
                             "line": None, "params": {}, "n_args": 0, "network": True})
    metrics = sorted({q.rsplit(".", 1)[1] for q in v.calls if q.startswith("sklearn.metrics.")}
                     | {c["name"] for c in v.components if c["kind"] == "metric"})
    return {
        "ok": True,
        "size": size,
        "structure": {**{k: v.structure.get(k, 0) for k in
                         ("functions", "classes", "for", "while", "if", "try", "with",
                          "comprehensions", "lambdas")},
                      "max_depth": v.max_depth, "complexity": v.complexity},
        "imports": sorted(v.imports),
        "components": v.components,
        "metrics": metrics,
        "data": {
            "reads": v.reads, "writes": v.writes,
            "column_writes": v.col_writes, "loc_writes": v.loc_writes, "inplace": v.inplace,
            "columns_written": len(v.columns_written),
            # by name only, so only for a file that imports the library at all
            # (gc.collect() is not polars' collect)
            "pandas": dict(v.pandas.most_common()) if "pandas" in v.imports else {},
            "polars": dict(v.polars.most_common()) if "polars" in v.imports else {},
        },
        "other": {"pip_installs": sorted(set(v.pip)), "shell_calls": v.shell, "gpu": v.gpu,
                  "seeds": sorted(v.seeds), "prints": v.prints},
    }


_TOKEN = re.compile(r"\S+")


def normalized(src: str) -> str | None:
    """The code with comments, docstrings and formatting removed (None if it
    does not parse): two files with the same normalized form are the same code."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body.pop(0)
            if not body:
                body.append(ast.Pass())
    return ast.unparse(tree)
