# APTOS 2019 blindness detection data on GCS

Input root: `gs://mle-trajectories-data/aptos2019-blindness-detection/input`.
The original Kaggle split contains 3,662 labelled training images and 1,928
test images. Files: `train.csv`, `test.csv`, `sample_submission.csv`, and PNGs
under `train_images/` and `test_images/`. Target: `diagnosis` (grades 0-4);
ID: `id_code`; metric: quadratic weighted kappa. Hidden evaluation remains
with Kaggle. No custom split, image transformation or label changes are made.

Read directly with the existing agent reader ADC:

```python
import pandas as pd
import fsspec
from PIL import Image

root = "gs://mle-trajectories-data/aptos2019-blindness-detection/input"
train = pd.read_csv(f"{root}/train.csv")
image_id = train.loc[0, "id_code"]
with fsspec.open(f"{root}/train_images/{image_id}.png", "rb") as stream:
    image = Image.open(stream).convert("RGB")
```

The dataset is ready only when `cloud_manifest.json` records `verified: true`
for the complete file inventory, sizes, CRC32C checksums and object generations.
Local `get_data.sh` retains the acquisition path for historical trajectories.
