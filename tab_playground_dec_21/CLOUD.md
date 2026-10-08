# December 2021 forest cover data on GCS

Input root: `gs://mle-trajectories-data/tab_playground_dec_21/input`.
The original Kaggle split contains 4,000,000 labelled rows and 1,000,000 test
rows. Train is stored as `train.csv.gz` (gzip of the unchanged original CSV);
`test.csv` and `sample_submission.csv` are unchanged. Target: `Cover_Type`;
ID: `Id`; metric: accuracy. Hidden evaluation remains with Kaggle.

This is the synthetic December 2021 competition, not the small classic UCI
CoverType dataset. There is no meaningful time axis. Keep rare-class handling
in mind when validating: class 5 has only one training row; class 4 has 377.
The training labels and IDs have not been filtered or changed.

```python
import pandas as pd
root = "gs://mle-trajectories-data/tab_playground_dec_21/input"
train = pd.read_csv(f"{root}/train.csv.gz")
test = pd.read_csv(f"{root}/test.csv")
```

Use the existing reader ADC. `cloud_manifest.json` records verified object
sizes, CRC32C checksums and generations. Local `get_data.sh` still acquires the
original uncompressed CSVs for historical trajectories.
Source: [Kaggle](https://www.kaggle.com/competitions/tabular-playground-series-dec-2021/data).
