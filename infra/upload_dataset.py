#!/usr/bin/env python3
"""Upload only agent-visible inputs using the selected maintainer gcloud profile."""
import argparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from google.api_core.exceptions import PreconditionFailed
from gcs_client import client as make_client, upload_file


TABULAR = ("train.csv", "test.csv", "sample_submission.csv")
FILES = {
    "aptos2019-blindness-detection": (*TABULAR, "train_images", "test_images"),
    "house_price": ("train.parquet", "test.csv", "sample_submission.csv"),
    "nyc_taxi_fare": TABULAR,
    "playground-series-s6e7": TABULAR,
    "tab_playground_dec_21": TABULAR,
    "ttt-task": (
        "tracking_graph_train.parquet", "target.tsv", "trackers.tsv",
        "domains.parquet", "link-graph.parquet", "url-classification.csv",
        "freedom-of-the-press.csv",
    ),
}
REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=FILES)
    parser.add_argument("--input", type=Path, help="Actual input folder, including symlinked inputs")
    parser.add_argument("--bucket", default="gs://mle-trajectories-data")
    parser.add_argument("--prefix", help="Dataset prefix, e.g. house_price/v1 (default: dataset name)")
    parser.add_argument("--gzip-train", action="store_true", help="Publish train.csv.gz instead of train.csv; rows are unchanged")
    parser.add_argument("--workers", type=int, default=8, help="Parallel large-file parts (1-32)")
    parser.add_argument("--resume-train-parts", help="Resume a known interrupted _uploads/<32-hex-id> prefix; each part is checksum-checked")
    parser.add_argument("--upload", action="store_true", help="Execute the upload; default is a dry run")
    args = parser.parse_args()
    source = args.input or REPO / args.dataset / "input"
    destination = f"{args.bucket.rstrip('/')}/{args.prefix or args.dataset}/input"
    if not args.bucket.startswith("gs://"):
        parser.error("--bucket must be a gs:// URI")
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be between 1 and 32')
    if args.resume_train_parts and not re.fullmatch(r'_uploads/[0-9a-f]{32}', args.resume_train_parts):
        parser.error('--resume-train-parts must identify a single uploader staging prefix')
    files = FILES[args.dataset]
    if args.gzip_train:
        if 'train.csv' not in files or args.dataset == 'aptos2019-blindness-detection':
            parser.error('--gzip-train is for tabular Kaggle inputs')
        files = tuple('train.csv.gz' if name == 'train.csv' else name for name in files)

    # Check the complete allowlist before uploading anything. Never recurse over
    # the dataset root: scoring/, raw/, credentials and agent runs stay private.
    missing = [name for name in files if not (source / name).exists()]
    if missing:
        parser.error(f"Missing inputs in {source}: {', '.join(missing)}")
    for name in files:
        path = source / name
        if path.is_dir():
            entries = list(path.rglob('*'))
            if not any(entry.is_file() for entry in entries):
                parser.error(f"Empty input directory: {path}")
            if any(entry.is_symlink() and not entry.exists() for entry in entries):
                parser.error(f"Broken symlink in {path}")
        elif not path.is_file() or path.stat().st_size == 0:
            parser.error(f"Empty or invalid input: {path}")

    paths = []
    for name in files:
        path = source / name
        paths.extend(sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path])
    if args.upload:
        client = make_client()
        bucket_name, prefix = destination[5:].split('/', 1)
        bucket = client.bucket(bucket_name)
        def upload(path):
            name = str(path.relative_to(source))
            try:
                uploaded = upload_file(bucket, f"{prefix}/{name}", path, workers=args.workers,
                                       resume_prefix=args.resume_train_parts if name.startswith('train.') else None)
            except PreconditionFailed:
                # Verification is separate and rejects existing bytes that differ.
                return f"Already present: {name}"
            return f"{'Uploaded' if uploaded else 'Already present'}: {name}"
        with ThreadPoolExecutor(max_workers=8) as pool:
            for count, result in enumerate(pool.map(upload, paths), 1):
                if len(paths) < 20 or count % 100 == 0 or count == len(paths):
                    print(f"{count}/{len(paths)} {result}", flush=True)
    else:
        print(f"Would upload {len(paths)} files ({sum(p.stat().st_size for p in paths)/1e9:.3f} GB) to {destination}")
    print(f"{'Uploaded' if args.upload else 'Dry run for'}: {destination}")
    if args.upload:
        print("Verify the published inputs before setting dataset.toml's data URI.")


if __name__ == "__main__":
    main()
