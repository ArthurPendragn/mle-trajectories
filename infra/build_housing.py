#!/usr/bin/env python3
"""Build UK housing v1 with a temporal test and a separate past-period diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import polars as pl

REPO = Path(__file__).resolve().parents[1]
CONFIG = {"house_price": ("id", "price", "rmse_log10")}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def housing(source, seed, private):
    names = {
        "Price": "price", "Property Type": "property_type", "Old/New": "is_new_build",
        "Duration": "tenure", "Town/City": "town", "District": "district",
        "County": "county", "PPDCategory Type": "sale_category",
    }
    raw = pl.scan_csv(source)
    guid = "Transaction unique identifier"
    stats = raw.select(
        pl.len().alias("rows"), pl.col(guid).n_unique().alias("unique_ids"),
        pl.col("Price").min().alias("min_price"),
        pl.col("Date of Transfer").min().alias("min_date"),
        pl.col("Date of Transfer").max().alias("max_date"),
    ).collect(engine="streaming").row(0, named=True)
    if (stats['rows'] != 22_489_348 or stats['unique_ids'] != stats['rows']
            or stats['min_price'] <= 0 or stats['min_date'][:10] != '1995-01-01'
            or stats['max_date'][:10] != '2017-06-29'):
        raise ValueError("Source does not match the verified v1 historical snapshot")
    raw = raw.with_columns(
        (pl.lit("R") + pl.col(guid).hash(seed=seed).cast(pl.String)).alias("id"),
        pl.col("Date of Transfer").str.slice(0, 10).alias("date"),
        (pl.col(guid).hash(seed=seed ^ 0x9E3779B9) % 20 == 0).alias("_past_holdout"),
    ).rename(names)
    cols = ["id", "price", "date", *[c for c in names.values() if c != "price"]]
    # Store the transformed full table outside the agent checkout. Subsequent
    # scans avoid repeatedly parsing the 2.4 GB source CSV.
    staging = private / "source.parquet"
    raw.select(*cols, "_past_holdout").sink_parquet(staging, compression="zstd")
    df = pl.scan_parquet(staging)
    future = pl.col("date") >= "2017-01-01"
    history = (pl.col("date") >= "2014-01-01") & ~future & pl.col("_past_holdout")
    train_path = private / "train_staging.parquet"
    df.filter(~future & ~history).drop("_past_holdout").sink_parquet(train_path, compression="zstd")
    test = (df.filter(future | history).drop("_past_holdout")
            .with_columns(pl.when(future).then(pl.lit("future")).otherwise(pl.lit("history")).alias("_slice"))
            .collect(engine="streaming").sort("id"))
    return train_path, test, {
        "strategy": "train through 2016; primary test January-June 2017; diagnostic 5% of 2014-2016 sales removed from training",
        "primary_slice": "future", "diagnostic_slice": "history",
        "source_rows": df.select(pl.len()).collect().item(),
        "limitation": "Curated historical snapshot, not a historical registry-vintage simulation; registration dates unavailable",
    }

def write_task(out, dataset, public):
    id_col, target, metric = CONFIG[dataset]
    root = f"gs://mle-trajectories-data/{dataset}/v1/input"
    details = "Predict sale price in GBP using sale date, property type, new-build flag, tenure and town/district/county. The source covers England and Wales. Train contains sales from 1995-2016. The primary score uses January-June 2017 sales; a separate diagnostic uses withheld 2014-2016 sales. Validate forecasting with forward time splits inside training. No future price observations or external labelled price data may be used. Report raw prices; the grader clips predictions to at least GBP 1 before log10."
    text = f"""# {dataset} — benchmark v1

{details}

Data root: `{root}`. Read GCS directly with the existing agent reader ADC;
`pandas.read_parquet`/`pandas.read_csv` with gcsfs support `gs://` URIs.

- Training file: `train.parquet` ({public['train_rows']:,} rows), including `{target}`.
- Test file: `test.csv` ({public['test_rows']:,} rows), without `{target}`.
- Submission: CSV columns `{id_col},{target}` with exactly one prediction for every test ID.
- Metric: `{metric}`. {'Lower is better.' if metric.startswith('rmse') else 'Higher is better.'}
- `sample_submission.csv` shows the required format; its predictions are a simple training-only baseline.

IDs are opaque identifiers, not features. Hidden labels are available only to
the maintainer grader. Use only supplied training labels for learning; do not
retrieve labels from the upstream public datasets, other runs or external sources.
Model fitting and validation must not use hidden-test feedback. The dataset
schema and split description are in `manifest.json` beside this task.
"""
    (out / "TASK.md").write_text(text)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    args.dataset = "house_price"
    private_root = args.private_root.resolve()
    if private_root.is_relative_to(REPO):
        parser.error("Private labels must be outside the repository tree")
    private = private_root / args.dataset / "v1"
    out = REPO / args.dataset / "benchmark_v1"
    if (out / "manifest.json").exists():
        parser.error("v1 already built; publish a new version to change the split")
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    inputs = REPO / args.dataset / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    seed_path = private / "seed.json"
    if seed_path.exists():
        seed = json.loads(seed_path.read_text())["seed"]
    else:
        seed = secrets.randbits(32)
        seed_path.write_text(json.dumps({"seed": seed, "polars_version": pl.__version__}))
        seed_path.chmod(0o600)
    id_col, target, metric = CONFIG[args.dataset]
    train_path, test, split = housing(args.source, seed, private)
    train = pl.scan_parquet(train_path)
    train.sink_parquet(inputs / "train.parquet", compression="zstd")
    labels = test.select(id_col, target, "_slice")
    labels.write_parquet(private / "test_labels.parquet", compression="zstd")
    public_test = test.drop(target, "_slice")
    public_test.write_csv(inputs / "test.csv")
    baseline = train.select(pl.col(target).median()).collect().item()
    public_test.select(id_col).with_columns(pl.lit(baseline).alias(target)).write_csv(inputs / "sample_submission.csv")
    stats = train.select(pl.len().alias("rows"), pl.col(id_col).n_unique().alias("unique")).collect()
    assert stats['rows'][0] == stats['unique'][0]
    assert labels[id_col].n_unique() == labels.height
    overlap = train.select(id_col).join(public_test.select(id_col).lazy(), on=id_col, how="inner").select(pl.len()).collect().item()
    assert overlap == 0, f"Train/test ID overlap: {overlap}"
    public = {
        "dataset": args.dataset, "version": "v1", "target": target, "id_column": id_col,
        "metric": metric, "train_rows": stats['rows'][0], "test_rows": test.height,
        "test_slices": dict(labels['_slice'].value_counts().iter_rows()),
        "split": split, "schema": {k: str(v) for k, v in train.collect_schema().items()},
        "files": {f"input/{p.name}": {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(inputs.iterdir()) if p.is_file()},
        "train_test_id_overlap": overlap,
    }
    (out / "manifest.json").write_text(json.dumps(public, indent=2) + '\n')
    (private / "manifest.json").write_text(json.dumps({
        **public, "source": str(args.source), "source_sha256": sha256(args.source),
        "source_bytes": args.source.stat().st_size, "seed": seed,
        "polars_version": pl.__version__,
    }, indent=2) + '\n')
    write_task(out, args.dataset, public)
    print(json.dumps({k: public[k] for k in ('dataset', 'train_rows', 'test_rows', 'test_slices', 'split')}, indent=2), flush=True)

if __name__ == "__main__":
    main()
