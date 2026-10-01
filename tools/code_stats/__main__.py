"""Print the static features of pipeline scripts as JSON.

    python -m code_stats path/to/pipeline.py [more.py ...]
    python -m code_stats parent.py child.py --diff
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import analyze_source
from .diff import compare


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="code_stats", description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--diff", action="store_true", help="compare the first file (parent) with the second")
    args = ap.parse_args(argv)
    srcs = [f.read_text(errors="replace") for f in args.files]
    if args.diff:
        if len(srcs) != 2:
            ap.error("--diff takes exactly two files")
        out = compare(analyze_source(srcs[0]), analyze_source(srcs[1]), srcs[0], srcs[1])
    else:
        out = {str(f): analyze_source(s) for f, s in zip(args.files, srcs)}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
