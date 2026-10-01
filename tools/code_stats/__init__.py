"""Static code features of pipeline scripts (stdlib only): size, structure,
ML components and their hyperparameters, data handling; and what changed
between a step and its parent. See README.md."""
from .analyze import analyze_source, normalized
from .catalog import KINDS, classify
from .diff import compare

__all__ = ["KINDS", "analyze_source", "classify", "compare", "normalized"]
