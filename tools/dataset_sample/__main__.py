"""Build a sample of a dataset from the [sample] recipe in its dataset.toml.

    python -m dataset_sample tab_playground_dec_21 --size 100k        # -> sample_100k/
    python -m dataset_sample tab_playground_dec_21 --size 100k --dry-run
    python -m dataset_sample ttt-task --size 200k --force             # rebuild
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import SampleError, build, parse_size, plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dataset_sample", description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path, help="dataset folder (holding dataset.toml and input/)")
    ap.add_argument("--size", required=True, help="rows for the sampled tables: 100k, 1m, 250000")
    ap.add_argument("--name", default=None, help="folder name (default: sample_<size>)")
    ap.add_argument("--force", action="store_true", help="replace an existing sample folder")
    ap.add_argument("--dry-run", action="store_true", help="print what would be built, as JSON")
    args = ap.parse_args(argv)
    try:
        size = parse_size(args.size)
        if args.dry_run:
            print(json.dumps(plan(args.dataset, size, args.name), indent=1))
        else:
            build(args.dataset, size, name=args.name, force=args.force)
    except SampleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
