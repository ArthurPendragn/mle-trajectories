#!/usr/bin/env python3
"""Verify uploaded dataset bytes against GCS size and CRC32C metadata."""
import argparse
import base64
import json
from pathlib import Path

import google_crc32c
from gcs_client import client as make_client

from upload_dataset import FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=FILES)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--prefix', required=True, help='Full gs:// input prefix')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gzip-train', action='store_true')
    args = parser.parse_args()
    if not args.prefix.startswith('gs://'):
        parser.error('--prefix must be gs://')
    bucket, prefix = args.prefix[5:].split('/', 1)
    # Use the selected gcloud maintainer profile without changing the agent ADC.
    client = make_client()
    remote = {blob.name.removeprefix(prefix.rstrip('/') + '/'): blob
              for blob in client.list_blobs(bucket, prefix=prefix.rstrip('/') + '/')}
    records = {}
    files = FILES[args.dataset]
    if args.gzip_train:
        files = tuple('train.csv.gz' if name == 'train.csv' else name for name in files)
    for name in files:
        path = args.input / name
        paths = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        if not paths:
            raise ValueError(f'No inputs found in {path}')
        for p in paths:
            relative = str(p.relative_to(args.input))
            checksum = google_crc32c.Checksum()
            with p.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    checksum.update(chunk)
            crc = base64.b64encode(checksum.digest()).decode()
            blob = remote.get(relative)
            if blob is None or blob.size != p.stat().st_size or blob.crc32c != crc:
                raise ValueError(f'Upload missing or checksum/size mismatch: {relative}')
            records[relative] = {'bytes': blob.size, 'crc32c': crc, 'generation': blob.generation}
    if set(records) != set(remote):
        raise ValueError(f'Unexpected cloud objects: {set(remote) - set(records)}')
    report = {'dataset': args.dataset, 'data': args.prefix, 'verified': True,
              'files': records, 'total_bytes': sum(r['bytes'] for r in records.values())}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Verified {len(records)} files, {report['total_bytes'] / 1e9:.3f} GB at {args.prefix}")


if __name__ == '__main__':
    main()
