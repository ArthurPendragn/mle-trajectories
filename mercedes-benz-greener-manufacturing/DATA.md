# mercedes-benz-greener-manufacturing — where the data comes from

[Kaggle Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing).
The competition's public test file has no labels, so `./get_data.sh` downloads
the competition and splits its labeled `train.csv` (4209 rows) into a new train
and test. That is what gives this task a test set that can be scored. Kaggle's
own test file is not used.

The recipe is `[prepare]` in `dataset.toml`: a row holdout on `y`,
`test_size` 0.2, seed 42. Regression, so the split is not stratified.

Pipelines read `./input/`. `input/` is gitignored. The competition download
itself stays in `raw/`, which is also gitignored.

| file | rows | `y` |
| --- | --- | --- |
| `input/train.csv` | 3367 | kept |
| `input/test.csv` | 842 | removed |

`scoring/answer.csv` is the removed `y` column, one value per test row,
in the same order. It is gitignored, so the agent never sees it.
