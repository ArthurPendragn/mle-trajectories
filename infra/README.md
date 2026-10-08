# Corpus data on GCS

Agent-visible bucket: `gs://mle-trajectories-data`, project `mle-agents`, region
`europe-west10`. Maintainer-only labels: `gs://mle-trajectories-private`.
Both use uniform bucket-level access and public access prevention.
The existing `nyc-lake-agent-reader@mle-agents.iam.gserviceaccount.com` has
`roles/storage.objectViewer` on the agent bucket, no binding on the private
bucket, and no project-level IAM role. The bucket and project IAM policies
have been inspected. This Mac's maintainer login cannot impersonate the reader;
use the existing reader profile on the agent machine.

| Dataset | Input URI | Evaluation | Status |
| --- | --- | --- | --- |
| house_price | `gs://mle-trajectories-data/house_price/v1/input` | Custom temporal test; private labels | Verified |
| nyc_taxi_fare | `gs://mle-trajectories-data/nyc_taxi_fare/input` | Original Kaggle test | Verified |
| playground-series-s6e7 | `gs://mle-trajectories-data/playground-series-s6e7/input` | Original Kaggle test | Verified |
| tab_playground_dec_21 | `gs://mle-trajectories-data/tab_playground_dec_21/input` | Original Kaggle test | Verified |
| aptos2019-blindness-detection | `gs://mle-trajectories-data/aptos2019-blindness-detection/input` | Original Kaggle test | Verified |

Publication is complete only when the dataset has `cloud_manifest.json` with
`verified: true`; the website's `dataset.toml` data URI is updated after that
verification. The four Kaggle datasets retain their original train/test splits.
December 2021 and NYC taxi publish `train.csv.gz`; S6E7 publishes `train.csv`. Gzip changes
only storage encoding. All use the original `test.csv` and `sample_submission.csv`.
APTOS publishes the original CSVs plus 3,662 training and 1,928 test PNGs under
`train_images/` and `test_images/` (5,593 files, 10.217 GB total).
NYC housing keeps its existing lake. TrackTheTrackers is deferred; BEAVER and
multi-table CoverType are excluded.

## Publish and verify

The scripts use the selected gcloud maintainer profile to authenticate the
native Cloud Storage SDK. They do not alter Python ADC or the agent profile.
On the GPU node the existing maintainer profile is selected by
`CLOUDSDK_CONFIG="$HOME/.config/gcloud-maint"`. Keep it away from agents.

```bash
uv sync
# Default is a local dry run, printing the upload size and destination.
uv run python infra/upload_dataset.py playground-series-s6e7
uv run python infra/upload_dataset.py playground-series-s6e7 --upload
uv run python infra/verify_upload.py playground-series-s6e7 \
  --input playground-series-s6e7/input \
  --prefix gs://mle-trajectories-data/playground-series-s6e7/input \
  --out playground-series-s6e7/cloud_manifest.json
```

`--gzip-train` selects `train.csv.gz` in both the uploader and verifier.
`--input` accepts another source directory; `--prefix house_price/v1` selects
housing's versioned destination. Only explicitly listed input files are
uploaded, never an entire dataset/run/scoring directory. Existing objects are
kept, and verification rejects mismatching bytes rather than overwriting them.
The uploader validates transfer CRC32C; the verifier independently compares
local CRC32C, sizes and the complete cloud file inventory. Input file symlinks
are followed. Create a new version prefix to change a published dataset.

The native SDK is used because gcloud's concurrent CLI uploads stalled on this
Mac. Large files are uploaded as parallel small objects, composed in order with
a final CRC32C check, and their temporary live objects are deleted. The CLI
remains the credential source and can be used for bucket inspection.
Interrupted large uploads retain completed parts. The uploader prints their
`_uploads/<id>` prefix; pass it as `--resume-train-parts` to resume a training
file with each retained part checked against the local bytes. Use `--workers 8`
on a slow uplink. Successful transfers remove their staging objects.

### NYC taxi publication

Verified on 2026-10-08: 55,423,856 labelled training rides and 9,914 test rides,
with the unchanged original Kaggle split. The three input files total
2,039,053,083 bytes. The resumed upload reused 188 checksum-validated parts;
successful composition and verification removed its staging objects.

To verify the published inputs against a local gzip copy:

```bash
uv run python infra/verify_upload.py nyc_taxi_fare --gzip-train \
  --input nyc_taxi_fare/input \
  --prefix gs://mle-trajectories-data/nyc_taxi_fare/input \
  --out nyc_taxi_fare/cloud_manifest.json
```

`CLOUD.md` and `cloud_manifest.json` are published under the dataset's cloud
root; `nyc_taxi_fare/dataset.toml` points to the verified input prefix. Direct
pandas reads of the gzip training CSV and test CSV passed with the reader
profile. Historical pipelines retain their original local CSV inputs.

## UK housing v1

The builder reads the full 22,489,348-row historical CSV provided from the
stratum benchmark folder. It preserves the eight predictor fields and original
prices, drops the original transaction GUID/record-status fields, and assigns
opaque IDs. All prices in this source are positive and transaction GUIDs unique.

- Training: 21,963,349 sales from 1995 through 2016.
- Primary hidden test: all 375,098 sales from January through June 2017.
- Separate diagnostic: 150,901 sales (5% of 2014-2016), removed from training.
- Metric: RMSE of log10 price; submit raw GBP, clipped to at least 1 by the grader.
- Scores are reported separately; the main score is the future test.

Use forward validation within the training period. The older slice measures
interpolation with the available pre-2017 history, not forecasting those older
sales from an earlier cutoff. It does not enter the primary score.
This is a curated transaction-date split, not a registry-vintage backtest:
registration timestamps are absent, and HM Land Registry describes reporting
lags in its [data guidance](https://www.gov.uk/guidance/about-the-price-paid-data).

The training-median baseline scores 0.4183 on the future test. A 2016-only
(`district`, `property_type`) log-mean lookup scores 0.2535, with 0.2095 on the
older diagnostic. These training-only baselines verify feasibility without
using test labels to choose the split or tune the model.

Agent task: `house_price/benchmark_v1/TASK.md` (also published at
`gs://mle-trajectories-data/house_price/v1/TASK.md`). Public schema/split/file
metadata: `house_price/benchmark_v1/manifest.json`. Private labels, seed and
provenance are at `gs://mle-trajectories-private/house_price/v1/`.
The seed and full labelled staging table stay outside this repository.

```bash
# Rebuild only in a fresh checkout/version; an existing v1 manifest is protected.
uv run python infra/build_housing.py --source /path/to/price_paid_records.csv \
  --private-root /path/outside/the/repo/private

# Maintainer grader: exactly one prediction per test ID is required.
uv run python infra/grade_housing.py submission.csv \
  --labels gs://mle-trajectories-private/house_price/v1/test_labels.parquet \
  --out score.json
```

The grader needs maintainer **ADC**, separately from CLI credentials. Private
labels can also be passed as a local parquet file outside the agent workspace.
Public source data cannot provide cryptographic label secrecy: agent runs must
use the supplied training labels and refrain from downloading upstream labels.

## Read directly in new agents

Keep the existing agent reader ADC from the NYC lake setup. A gcloud CLI login
alone does not configure Python ADC. On GCE, attach the reader service account;
elsewhere use the existing reader setup. Never use maintainer credentials in an
agent run. Python's gcsfs reads transfer bytes into memory without a persistent
local copy. Full CSV scans still transfer the whole CSV; Parquet supports
selective reads.

```python
import pandas as pd
DATA_ROOT = "gs://mle-trajectories-data/house_price/v1/input"
train = pd.read_parquet(f"{DATA_ROOT}/train.parquet")
test = pd.read_csv(f"{DATA_ROOT}/test.csv")
```

For a DataOps plan, record the URI and read operation:

```python
import pandas as pd
import stratum as skrub
DATA_ROOT = "gs://mle-trajectories-data/house_price/v1/input"
data = skrub.as_data_op(f"{DATA_ROOT}/train.parquet").skb.apply_func(pd.read_parquet)
```

Historical trajectory scripts are preserved. New runs should use the GCS
roots; moving data does not automatically rewrite old local-path pipelines.
Local `get_data.sh` scripts remain available to reproduce the historical runs.

References: [gcsfs](https://gcsfs.readthedocs.io/en/stable/) and
[Cloud Storage authentication](https://docs.cloud.google.com/storage/docs/authentication).

## Python environment

`uv sync` works on Apple Silicon macOS. `uv sync --group website` includes the
site backend. Torch uses PyPI native wheels outside Linux x86-64; Linux x86-64
retains the CUDA 12.6 index using
[uv platform source markers](https://docs.astral.sh/uv/concepts/indexes/).
