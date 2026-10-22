import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

TRAIN = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
SEED = 73191
SIZES = [50000, 200000, 800000, 2000000]
VALIDATION_SIZE = 30000


class FixedValidationSets(BaseEstimator):
    def __init__(self, validation_size=30000, n_splits=3):
        self.validation_size = validation_size
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        boundary = self.validation_size * self.n_splits
        train = np.arange(boundary, len(X))
        for fold in range(self.n_splits):
            start = fold * self.validation_size
            yield train, np.arange(start, start + self.validation_size)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def build():
    columns = [
        "fare_amount", "pickup_datetime", "pickup_longitude",
        "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
        "passenger_count",
    ]
    dtypes = {c: "float32" for c in columns if c != "pickup_datetime"}
    source = skrub.as_data_op(TRAIN).skb.apply_func(
        pd.read_csv, usecols=columns, dtype=dtypes
    )
    # Pandas samples positions without replacement across the entire source.
    # The shuffled sample order defines all nested training sets and holdouts.
    rows = source.sample(
        n=2000000 + 3 * VALIDATION_SIZE,
        random_state=SEED,
        replace=False,
    ).reset_index(drop=True)
    y = rows["fare_amount"]
    dt = rows["pickup_datetime"].skb.apply_func(
        pd.to_datetime, utc=True, errors="coerce"
    )
    coords = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
    ]
    features = rows[coords + ["passenger_count"]].copy()
    dx = (rows["dropoff_longitude"] - rows["pickup_longitude"]) * 84.3
    dy = (rows["dropoff_latitude"] - rows["pickup_latitude"]) * 111.0
    features = features.assign(
        dx=dx.clip(-500, 500),
        dy=dy.clip(-500, 500),
        euclidean_km=((dx ** 2 + dy ** 2) ** 0.5).clip(0, 1000),
        manhattan_km=(dx.abs() + dy.abs()).clip(0, 1000),
        year=dt.dt.year,
        month=dt.dt.month,
        dayofweek=dt.dt.dayofweek,
        hour=dt.dt.hour,
    )
    for airport, lon, lat in [
        ("jfk", -73.7781, 40.6413),
        ("lga", -73.8740, 40.7769),
        ("ewr", -74.1745, 40.6895),
    ]:
        for end in ["pickup", "dropoff"]:
            distance = (
                ((rows[end + "_longitude"] - lon) * 84.3) ** 2
                + ((rows[end + "_latitude"] - lat) * 111.0) ** 2
            ) ** 0.5
            features = features.assign(**{
                end + "_" + airport + "_km": distance.clip(0, 1000)
            })
    features = features.skb.set_name("geography_calendar")

    estimator = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=-999)),
        ("regressor", XGBRegressor(
            objective="reg:squarederror",
            device="cuda",
            tree_method="hist",
            n_estimators=600,
            max_depth=8,
            learning_rate=0.07,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=10.0,
            max_bin=256,
            n_jobs=8,
            random_state=SEED,
        )),
    ])
    curve = skrub.as_data_op(estimator).skb.apply_func(
        learning_curve,
        features,
        y,
        train_sizes=SIZES,
        cv=FixedValidationSets(VALIDATION_SIZE, 3),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        shuffle=False,
        return_times=True,
        error_score="raise",
    )
    sizes = curve[0]
    scores = curve[2]
    fit_times = curve[3]
    score_times = curve[4]
    total_times = (fit_times + score_times).skb.apply_func(np.sum, axis=1)
    gains = scores.skb.apply_func(np.diff, axis=0)
    gain_mean = gains.skb.apply_func(np.mean, axis=1)
    gain_se = gains.skb.apply_func(np.std, axis=1, ddof=1) / np.sqrt(3)
    return {
        "training_sizes": sizes,
        "validation_count_per_fold": skrub.as_data_op(VALIDATION_SIZE),
        "fold_rmse": -scores,
        "fold_train_rmse": -curve[1],
        "fold_fit_seconds": fit_times,
        "fold_score_seconds": score_times,
        "all_fold_seconds": total_times,
        "paired_rmse_improvement": gains,
        "paired_gain_mean": gain_mean,
        "paired_gain_standard_error": gain_se,
        "linear_runtime_ceiling_rows": (
            sizes[-1] * (28000.0 / 12.0) / total_times[-1]
        ),
        "documentation": skrub.as_data_op(
            "Population: unfiltered; no target clipping or target-based exclusions. "
            "Read all 55,423,856 source rows, then pandas sample without replacement "
            "with random_state=73191, n=2,090,000. This samples across the full "
            "source rather than taking a prefix. No materialized files are used. "
            "First 90,000 sampled rows form three disjoint 30,000-row validation "
            "sets. All training sets exclude all 90,000 validation rows; nested "
            "training prefixes of the remaining shuffled sample are reused in "
            "each fold. Validation retains every target value. "
            "Features: coordinates, passengers, clipped planar displacements, "
            "Euclidean and Manhattan distance, airport endpoint distances, "
            "year/month/weekday/hour; no identifier or timestamp frequency. "
            "Model: CUDA XGBoost squared-error, 600 trees, depth 8, lr .07. "
            "Timing includes per-fold imputation, fitting, training-set scoring "
            "and validation scoring from sklearn learning_curve. Shared CSV "
            "reading, sampling and graph-based feature construction are excluded; "
            "therefore the reported runtime ceiling is optimistic, not a measured "
            "end-to-end experiment ceiling. Positive paired gains mean lower RMSE. "
            "Choose the largest affordable measured size unless the final paired "
            "gain is indistinguishable from zero; a still-positive final gain "
            "supports investigating a larger pool, subject to runtime overhead. "
            "The linear ceiling extrapolates the largest measured all-fold time "
            "to twelve experiments in 28,000 seconds."
        ),
    }