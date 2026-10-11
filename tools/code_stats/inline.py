"""Inline what a script uses from the run's own shared modules.

Agents that keep their code in shared modules (mle-claude: ``common.py``,
``features.py``, ``nn.py``) leave each pipeline script a few lines long. To
measure such a script like a self-contained one, ``inline_local`` appends the
source of every top-level definition the script reaches in those modules:
the names it imports, then, transitively, the names those definitions use
(within a module, and across modules through their own local imports). The
modules' third-party imports come along, so components resolve as usual.

Each definition is inlined once, however often it is called: a second call
runs the same code, it does not write more of it. Static and conservative:
a whole function counts, including branches its arguments switch off.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Module:
    src: str
    defs: dict[str, ast.stmt] = field(default_factory=dict)    # top-level name -> statement
    order: dict[int, int] = field(default_factory=dict)        # id(stmt) -> position
    imports: list[ast.stmt] = field(default_factory=list)      # third-party / stdlib imports
    local: dict[str, tuple[str, str]] = field(default_factory=dict)   # name -> (module, name)
    aliases: dict[str, str] = field(default_factory=dict)      # ``import common as c``: c -> common


def _local_imports(tree: ast.Module, local: set[str]):
    """(bound name, module, name or None for ``import module``) of local imports."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in local:
            for a in node.names:
                if a.name != "*":
                    yield a.asname or a.name, node.module, a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in local:
                    yield a.asname or a.name, a.name, None


def _parse(src: str, local: set[str]) -> _Module | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    m = _Module(src)
    for i, node in enumerate(tree.body):
        m.order[id(node)] = i
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            if not any(md in local for md in mods if md):
                m.imports.append(node)
        for n in names:
            m.defs[n] = node
    for bound, mod, name in _local_imports(tree, local):
        if name is None:
            m.aliases[bound] = mod
        else:
            m.local[bound] = (mod, name)
    return m


def _uses(node: ast.AST, m: _Module) -> set[tuple[str | None, str]]:
    """Names ``node`` refers to: (None, name) in its own module, (module, name) elsewhere."""
    out: set[tuple[str | None, str]] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            if n.id in m.local:
                out.add(m.local[n.id])
            elif n.id in m.defs:
                out.add((None, n.id))
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in m.aliases:
            out.add((m.aliases[n.value.id], n.attr))
    return out


def inline_local(src: str, modules: dict[str, str]) -> tuple[str, dict[str, list[str]]]:
    """``src`` with the definitions it reaches in ``modules`` (name -> source)
    appended, and which names were inlined per module."""
    local = set(modules)
    script = _parse(src, local)
    if script is None or not (script.local or script.aliases):
        return src, {}
    parsed: dict[str, _Module | None] = {}

    def mod(name: str) -> _Module | None:
        if name not in parsed:
            parsed[name] = _parse(modules[name], local) if name in modules else None
        return parsed[name]

    reached: dict[str, set[int]] = {}           # module -> ids of reached statements
    names: dict[str, set[str]] = {}
    # the script's own names stay where they are; only the shared modules' are followed
    work = [(md, n) for md, n in _uses(ast.parse(src), script) if md is not None]
    while work:
        md, n = work.pop()
        m = mod(md)
        if m is None or n not in m.defs:
            continue
        node = m.defs[n]
        names.setdefault(md, set()).add(n)
        if id(node) in reached.setdefault(md, set()):
            continue
        reached[md].add(id(node))
        for umd, un in _uses(node, m):
            work.append((umd or md, un))

    parts = [src.rstrip("\n")]
    for md in sorted(reached):
        m = parsed[md]
        stmts = sorted({id(s): s for s in m.defs.values() if id(s) in reached[md]}.values(),
                       key=lambda s: m.order[id(s)])
        body = [ast.get_source_segment(m.src, s, padded=False) or "" for s in m.imports + stmts]
        parts.append(f"\n\n# ---- inlined from {md}.py ----\n" + "\n\n".join(body))
    return "\n".join(parts) + "\n", {md: sorted(v) for md, v in sorted(names.items())}
