"""What changed between a step's code and its parent's, from their features
and their text: lines, components, hyperparameters, imports, data read."""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict

from .analyze import normalized

#: component kinds worth reporting as added / removed between two steps
REPORTED = ("model", "ensemble", "transformer", "pipeline", "splitter", "search", "metric")


def _components(f: dict) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in f.get("components") or []:
        if c["kind"] in REPORTED:
            out[(c["kind"], c["name"])].append(c)
    return out


def compare(parent: dict, child: dict, parent_src: str, child_src: str,
            normalized_pair: tuple[str | None, str | None] | None = None) -> dict:
    """``normalized_pair`` passes precomputed ``normalized()`` forms (a caller
    comparing many pairs parses each file once instead of once per pair)."""
    a, b = parent_src.splitlines(), child_src.splitlines()
    sm = difflib.SequenceMatcher(None, a, b)
    added = removed = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "delete"):
            removed += i2 - i1
        if op in ("replace", "insert"):
            added += j2 - j1
    na, nb = normalized_pair or (normalized(parent_src), normalized(child_src))

    pc, cc = _components(parent), _components(child)
    count_p = Counter({k: len(v) for k, v in pc.items()})
    count_c = Counter({k: len(v) for k, v in cc.items()})
    fmt = lambda n, m: n if m == 1 else f"{n} ×{m}"
    comp_added = sorted(fmt(n, m) for (k, n), m in (count_c - count_p).items())
    comp_removed = sorted(fmt(n, m) for (k, n), m in (count_p - count_c).items())

    # hyperparameters: the i-th occurrence of a component against the i-th in the parent
    params = []
    for key in sorted(set(pc) & set(cc)):
        for old, new in zip(pc[key], cc[key]):
            for p in sorted(set(old["params"]) | set(new["params"])):
                o, n = old["params"].get(p, "—"), new["params"].get(p, "—")
                # a parameter missing on a side that passes **kwargs may well be
                # in them: not reported as removed/added, only the ** change is
                if (o == "—" and "**" in old["params"]) or (n == "—" and "**" in new["params"]):
                    continue
                if o != n:
                    params.append({"component": key[1], "param": p, "old": o, "new": n})

    ia, ib = set(parent.get("imports") or ()), set(child.get("imports") or ())
    ra = {r["path"] for r in (parent.get("data") or {}).get("reads", []) if r["path"]}
    rb = {r["path"] for r in (child.get("data") or {}).get("reads", []) if r["path"]}
    loc = lambda f: (f.get("size") or {}).get("loc", 0)
    return {
        "same_code": na is not None and na == nb,
        "similarity": round(sm.ratio(), 3),
        "lines_added": added, "lines_removed": removed,
        "loc_delta": loc(child) - loc(parent),
        "components_added": comp_added, "components_removed": comp_removed,
        "params_changed": params,
        "imports_added": sorted(ib - ia), "imports_removed": sorted(ia - ib),
        "reads_added": sorted(rb - ra), "reads_removed": sorted(ra - rb),
    }
