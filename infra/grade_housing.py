#!/usr/bin/env python3
"""Maintainer-only UK housing grader; the primary score is future RMSE(log10)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def grade(submission, labels):
    if set(submission.columns) != {"id", "price"}:
        raise ValueError("Submission must contain exactly id,price")
    if submission['id'].isna().any() or submission['id'].duplicated().any():
        raise ValueError("Submission IDs must be non-null and unique")
    if labels['id'].duplicated().any():
        raise ValueError("Duplicate IDs in private labels")
    if set(submission['id']) != set(labels['id']):
        raise ValueError("Submission IDs must exactly match the hidden test IDs")
    scored = labels.merge(submission, on="id", validate="one_to_one", suffixes=("_true", "_pred"))
    predictions = pd.to_numeric(scored['price_pred'], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite numeric values")
    truth = scored['price_true'].to_numpy(dtype=float)
    if not np.isfinite(truth).all() or (truth <= 0).any():
        raise ValueError("Invalid private price labels")
    error = (np.log10(np.maximum(predictions, 1)) - np.log10(truth)) ** 2
    scores = {}
    for name in ("future", "history"):
        mask = scored['_slice'].eq(name).to_numpy()
        if not mask.any():
            raise ValueError(f"Missing required private slice: {name}")
        scores[name] = {"rows": int(mask.sum()), "rmse_log10": float(np.sqrt(error[mask].mean()))}
    return {"metric": "rmse_log10", "lower_is_better": True,
            "score": scores['future']['rmse_log10'], "primary_slice": "future", "slices": scores}


def read_frame(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, dtype={"id": str})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission")
    parser.add_argument("--labels", required=True, help="Private test_labels.parquet path or gs:// URI")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = grade(read_frame(args.submission), read_frame(args.labels))
    rendered = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
