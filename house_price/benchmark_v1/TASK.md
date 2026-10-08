# house_price — benchmark v1

Predict sale price in GBP using sale date, property type, new-build flag, tenure and town/district/county. The source covers England and Wales. Train contains sales from 1995-2016. The primary score uses January-June 2017 sales; a separate diagnostic uses withheld 2014-2016 sales. Validate forecasting with forward time splits inside training. No future price observations or external labelled price data may be used. Report raw prices; the grader clips predictions to at least GBP 1 before log10.

Data root: `gs://mle-trajectories-data/house_price/v1/input`. Read GCS directly with the existing agent reader ADC;
`pandas.read_parquet`/`pandas.read_csv` with gcsfs support `gs://` URIs.

- Training file: `train.parquet` (21,963,349 rows), including `price`.
- Test file: `test.csv` (525,999 rows), without `price`.
- Submission: CSV columns `id,price` with exactly one prediction for every test ID.
- Metric: `rmse_log10`. Lower is better.
- `sample_submission.csv` shows the required format; its predictions are a simple training-only baseline.

IDs are opaque identifiers, not features. Hidden labels are available only to
the maintainer grader. Use only supplied training labels for learning; do not
retrieve labels from the upstream public datasets, other runs or external sources.
Model fitting and validation must not use hidden-test feedback. The dataset
schema and split description are in `manifest.json` beside this task.
