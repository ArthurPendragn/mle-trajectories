import numpy as np
import pandas as pd
import skrub

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
SEED = 73191
POOL_SIZE = 16090000


class FixedDateHourSplits:
    def __init__(self, random_state=73191, validation_size=30000):
        self.random_state = random_state
        self.validation_size = validation_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return 3

    def split(self, X, y=None, groups=None):
        hours = X["pickup_datetime"].astype(str).str.slice(0, 13).to_numpy()
        unique, inverse, counts = np.unique(
            hours, return_inverse=True, return_counts=True
        )
        order = np.random.RandomState(self.random_state).permutation(len(unique))
        assignment = np.full(len(unique), -1, dtype=np.int8)
        start = 0
        for fold in range(3):
            remaining_counts = counts[order[start:]]
            width = int(
                np.searchsorted(
                    np.cumsum(remaining_counts), self.validation_size
                )
            ) + 1
            stop = min(start + width, len(order))
            assignment[order[start:stop]] = fold
            start = stop
        row_assignment = assignment[inverse]
        training_indices = np.flatnonzero(row_assignment == -1)
        for fold in range(3):
            yield training_indices.copy(), np.flatnonzero(row_assignment == fold)


def build():
    numeric = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "passenger_count"
    ]
    columns = ["key", "fare_amount", "pickup_datetime"] + numeric
    source = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply_func(
        pd.read_csv,
        usecols=columns,
        dtype={column: "float32" for column in numeric},
    )
    rows = source.sample(n=POOL_SIZE, random_state=SEED).reset_index(drop=True)
    y = rows["fare_amount"].skb.mark_as_y()
    keys = rows["key"]
    inputs = rows.drop(columns=["key", "fare_amount"])
    inputs = inputs.assign(
        pickup_datetime=inputs["pickup_datetime"].str.slice(0, 13)
        + ":00:00 UTC"
    )
    X = inputs.skb.mark_as_X(
        cv=FixedDateHourSplits(),
        split_kwargs={},
    )

    comparisons = [
        ("all", 250000, .7910, .5252),
        ("supported_years", 250000, .7910, .5252),
        ("supported_dates", 127987, .7809, .4951),
        ("month_weekday_hour", 152566, .7870, .4960),
        ("date_hour", 8244, .7422, .4991),
        ("hour_neighborhood", 49645, .7683, .4637),
        ("fare_ge_0", 249985, .7897, .5029),
        ("fare_ge_2_5", 249975, .7893, .5086),
        ("fare_ge_5", 215822, .7940, .5065),
        ("fare_le_100", 249916, .7944, .5263),
        ("fare_le_200", 249997, .7895, .5184),
        ("fare_0_100", 249901, .7918, .5128),
        ("fare_2_5_100", 249891, .7932, .5197),
        ("fare_2_5_200", 249972, .7937, .5210),
        ("observable_valid", 243879, .7861, .4961),
        ("valid_ge_2_5", 243858, .7880, .5077),
        ("valid_le_100", 243804, .7878, .4793),
        ("valid_2_5_100", 243783, .7841, .5053),
    ]
    metadata = {
        "population_comparisons": [
            {
                "rule": rule,
                "sample_count": count,
                "estimated_source_count": count * 55423856 / 250000,
                "ordinary_auc": ordinary,
                "date_hour_grouped_auc": grouped,
            }
            for rule, count, ordinary, grouped in comparisons
        ],
        "sampling": {
            "seed": SEED,
            "sample_size": POOL_SIZE,
            "method": "Full-source pandas sample without replacement; no target exclusions.",
            "identity_reuse": (
                "Row keys and deterministic split construction are retained in the "
                "locked graph; no external manifests or generated files are inputs."
            ),
        },
        "validation": (
            "Seeded permutation of whole date-hours; accumulate at least 30000 "
            "rows for each of three disjoint validation subsets. All validation "
            "hours are excluded from the common training pool. Timestamp inputs "
            "are coarsened to hours. Validation retains every target value."
        ),
        "population_choice": (
            "Keep all rows. Grouped all-row AUC folds were "
            "[0.559139,0.509440,0.506871]. Ordinary AUC indicates substantial "
            "timestamp-cluster separation; grouped results do not prove equality. "
            "Temporal restrictions remove 39–97 percent without predictive evidence."
        ),
        "learning_curve": {
            "model": "CUDA XGBoost squared error, 600 trees, depth 8, learning_rate .07",
            "sizes": [800000, 2000000, 8000000],
            "fold_rmse": [
                [3.889151, 3.894376, 3.867271],
                [3.836177, 3.804318, 3.810157],
                [3.783011, 3.784288, 3.700720],
            ],
            "feature_inclusive_recurring_seconds": [27.746, 74.022, 357.680],
            "paired_gain_mean": [.066715, .060878],
            "paired_gain_standard_error": [.011732, .026096],
            "size_reason": (
                "No established plateau. Approximately 16 million training rows "
                "chosen by extrapolation: twice the measured 8m runtime, with "
                "threefold slowdown allowance, permits twelve experiments in "
                "approximately 25753 seconds plus preparation. Conditional "
                "runtime and memory estimate, not a measured guarantee."
            ),
        },
    }
    audit = {
        "source_count": source.shape[0],
        "sample_count": rows.shape[0],
    }
    for name, value in metadata.items():
        audit[name] = skrub.as_data_op(value)

    return {
        "X": X,
        "y": y,
        "scoring": "neg_root_mean_squared_error",
        "row_keys": keys,
        "audit": audit,
    }