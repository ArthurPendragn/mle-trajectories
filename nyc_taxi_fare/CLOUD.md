# NYC taxi fare data on GCS

Upload is paused as of 2026-10-08; the training object is not yet published.
See [resume instructions](../infra/README.md#paused-nyc-taxi-upload). This
dataset is ready only after `cloud_manifest.json` confirms verification.

Input root: `gs://mle-trajectories-data/nyc_taxi_fare/input`.
The original split has 55,423,856 labelled rides (2009-2015) and 9,914 test rides.
Files: `train.csv.gz` (gzip of the unchanged original labelled CSV), original
`test.csv`, and original `sample_submission.csv`. Target: `fare_amount` in USD;
ID: `key`; metric: RMSE. Hidden evaluation remains with Kaggle. No custom test
split, filtering, coordinate cleaning, or label changes were introduced.

Pickup timestamps, pickup/dropoff coordinates and passenger count are the
available predictors. The source contains anomalous coordinates, passenger
counts and fares; cleaning and geographic feature engineering remain part of
the agent's task. The large historical training set spans changing fare regimes.

```python
import pandas as pd
root = "gs://mle-trajectories-data/nyc_taxi_fare/input"
# Use chunks or a row cap while exploring a 55M-row CSV.
train = pd.read_csv(f"{root}/train.csv.gz", nrows=100_000)
test = pd.read_csv(f"{root}/test.csv")
```

Use the existing reader ADC. Streaming gzip avoids a persistent local copy,
but a full scan still transfers the compressed file. `cloud_manifest.json`
records verified sizes, CRC32C checksums and generations. Local `get_data.sh`
retains the original acquisition path for historical trajectory scripts.
Source: [Kaggle](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data).
