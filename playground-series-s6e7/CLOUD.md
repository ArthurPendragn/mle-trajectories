# Student health risk data on GCS

Input root: `gs://mle-trajectories-data/playground-series-s6e7/input`.
Original Kaggle files: 690,088 labelled train rows, 295,753 test rows, and
`sample_submission.csv`. Target: `health_condition`; ID: `id`.
Metric: balanced accuracy. Test labels and grading remain with Kaggle.
No custom split or new labels were introduced. Local `get_data.sh` still works.

For a new agent, read CSVs directly with pandas and the existing reader ADC:

```python
import pandas as pd
root = "gs://mle-trajectories-data/playground-series-s6e7/input"
train = pd.read_csv(f"{root}/train.csv")
test = pd.read_csv(f"{root}/test.csv")
```

`cloud_manifest.json` records verified object sizes, CRC32C checksums and generations.
Source: [Kaggle](https://www.kaggle.com/competitions/playground-series-s6e7/data).
