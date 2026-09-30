"""Stdlib-only helpers shared with pipeline_analyzer.runtime (which must stay
importable without numpy): where a sample's manifest lives and how a sample's
data is identified."""
from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST = "sample_manifest.json"


def is_sample(folder: Path) -> bool:
    """A run-root built as a sample: ``input/`` plus a manifest beside it."""
    return (folder / "input").is_dir() and (folder / MANIFEST).is_file()


def fingerprint(folder: Path) -> str:
    """Identity of a sample's data: every file's path and size under input/.
    A rebuild from the same recipe, size and seed gives the same fingerprint;
    a different recipe or size almost surely does not."""
    h = hashlib.sha1()
    for p in sorted((Path(folder) / "input").rglob("*")):
        if p.is_file():
            h.update(f"{p.relative_to(folder)}\0{p.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def data_fingerprint(run_in: Path) -> str | None:
    """The fingerprint when ``run_in`` is a sample, None for the full data."""
    return fingerprint(run_in) if is_sample(Path(run_in)) else None
