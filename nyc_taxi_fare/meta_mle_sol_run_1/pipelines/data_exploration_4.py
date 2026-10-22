import time
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor

TRAIN = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
SIZES = [800000, 2000000, 8000000]
SOURCE_ROWS = 55423856

EXPR = """
dx = (dropoff_longitude - pickup_longitude) * 84.3
dy = (dropoff_latitude - pickup_latitude) * 111.0
dx = dx.clip(-100, 100)
dy = dy.clip(-100, 100)
euclidean = sqrt(dx ** 2 + dy ** 2)
manhattan = abs(dx) + abs(dy)
year = pickup_datetime.dt.year
month = pickup_datetime.dt.month
weekday = pickup_datetime.dt.dayofweek
hour = pickup_datetime.dt.hour
pickup_jfk = sqrt(((pickup_longitude + 73.7781) * 84.3) ** 2 + ((pickup_latitude - 40.6413) * 111.0) ** 2)
dropoff_jfk = sqrt(((dropoff_longitude + 73.7781) * 84.3) ** 2 + ((dropoff_latitude - 40.6413) * 111.0) ** 2)
pickup_lga = sqrt(((pickup_longitude + 73.8740) * 84.3) ** 2 + ((pickup_latitude - 40.7769) * 111.0) ** 2)
dropoff_lga = sqrt(((dropoff_longitude + 73.8740) * 84.3) ** 2 + ((dropoff_latitude - 40.7769) * 111.0) ** 2)
pickup_ewr = sqrt(((pickup_longitude + 74.1745) * 84.3) ** 2 + ((pickup_latitude - 40.6895) * 111.0) ** 2)
dropoff_ewr = sqrt(((dropoff_longitude + 74.1745) * 84.3) ** 2 + ((dropoff_latitude - 40.6895) * 111.0) ** 2)
"""

FEATURES = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude",
    "dropoff_latitude", "passenger_count", "dx", "dy",
    "euclidean", "manhattan", "year", "month", "weekday", "hour",
    "pickup_jfk", "dropoff_jfk", "pickup_lga", "dropoff_lga",
    "pickup_ewr", "dropoff_ewr"
]


class FixedValidation:
    def get_n_splits(self, X=None, y=None, groups=None):
        return 3

    def split(self, X, y=None, groups=None):
        train = np.arange(90000, len(X))
        for fold in range(3):
            yield train, np.arange(fold * 30000, (fold + 1) * 30000)


class TimedFareRegressor(RegressorMixin, BaseEstimator):
    """Fitted regressor with per-fold instrumentation for this exploration."""

    def fit(self, X, y):
        start = time.perf_counter()
        features = X.eval(EXPR, engine="python")[FEATURES]
        feature_seconds = time.perf_counter() - start

        start = time.perf_counter()
        self.imputer_ = SimpleImputer(strategy="median")
        numeric = self.imputer_.fit_transform(features)
        preprocess_seconds = time.perf_counter() - start

        self.model_ = XGBRegressor(
            objective="reg:squarederror",
            device="cuda",
            tree_method="hist",
            n_estimators=600,
            max_depth=8,
            learning_rate=0.07,
            n_jobs=8,
            random_state=73191,
            verbosity=0
        )
        start = time.perf_counter()
        self.model_.fit(numeric, y)
        model_seconds = time.perf_counter() - start
        print({
            "training_rows": len(X),
            "feature_seconds": feature_seconds,
            "preprocess_seconds": preprocess_seconds,
            "model_fit_seconds": model_seconds
        }, flush=True)
        return self

    def predict(self, X):
        start = time.perf_counter()
        features = X.eval(EXPR, engine="python")[FEATURES]
        feature_seconds = time.perf_counter() - start
        start = time.perf_counter()
        numeric = self.imputer_.transform(features)
        preprocess_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = self.model_.predict(numeric)
        prediction_seconds = time.perf_counter() - start
        print({
            "prediction_rows": len(X),
            "feature_seconds": feature_seconds,
            "preprocess_seconds": preprocess_seconds,
            "model_prediction_seconds": prediction_seconds
        }, flush=True)
        return prediction


def build():
    columns = [
        "fare_amount", "pickup_datetime", "pickup_longitude",
        "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
        "passenger_count"
    ]
    dtypes = {c: "float32" for c in columns if c != "pickup_datetime"}
    source = skrub.as_data_op(TRAIN).skb.apply_func(
        pd.read_csv, usecols=columns, dtype=dtypes
    )
    rows = source.sample(n=8090000, random_state=73191).reset_index(drop=True)
    rows = rows.assign(
        pickup_datetime=rows["pickup_datetime"].skb.apply_func(
            pd.to_datetime, utc=True, errors="coerce"
        )
    )
    X = rows.drop(columns=["fare_amount"])
    y = rows["fare_amount"]

    curve = skrub.as_data_op(TimedFareRegressor()).skb.apply_func(
        learning_curve, X, y,
        train_sizes=SIZES,
        cv=FixedValidation(),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        shuffle=False,
        return_times=True,
        error_score="raise"
    )
    rmse = -curve[2]
    gains = rmse[:-1] - rmse[1:]
    fit_seconds = curve[3]
    score_seconds = curve[4]
    totals = fit_seconds.sum(axis=1) + score_seconds.sum(axis=1)
    seconds_per_row = totals[-1] / SIZES[-1]
    conservative_ceiling = ((28000 - 300) / 12 / 3) / seconds_per_row
    ceiling = conservative_ceiling.skb.apply_func(
        np.clip, 0, SOURCE_ROWS - 90000
    )
    return {
        "training_sizes": curve[0],
        "fold_rmse": rmse,
        "paired_rmse_improvement": gains,
        "paired_gain_mean": gains.mean(axis=1),
        "paired_gain_standard_error": gains.std(axis=1, ddof=1) / np.sqrt(3),
        "fold_feature_preprocess_fit_seconds": fit_seconds,
        "fold_feature_preprocess_score_seconds": score_seconds,
        "all_fold_recurring_seconds": totals,
        "conservative_source_bounded_ceiling_rows": ceiling,
        "sample_memory_bytes": rows.memory_usage(deep=True).sum(),
        "documentation": skrub.as_data_op(
            "Full-source pandas sampling without replacement, seed 73191, "
            "8,090,000 rows. First 90,000 are three disjoint 30,000-row "
            "validation subsets; every training size uses the same nested "
            "prefix of the remaining shuffled rows. All labels are retained. "
            "The fitted estimator instruments feature construction, median "
            "imputation, model fitting and prediction separately in the logs. "
            "learning_curve times include these operations and validation RMSE "
            "scoring on every fold. No custom function transformer is used. "
            "CSV reading, sampling and timestamp parsing are shared preparation "
            "excluded from recurring times; harness wall timing includes them. "
            "No files are materialized. Preprocessing is explicitly defined "
            "here; exact identity with the earlier implementation is not "
            "established. The runtime ceiling reserves 300 seconds for "
            "preparation and uses a threefold slowdown allowance for twelve "
            "experiments. It is a source-bounded linear extrapolation, not a "
            "larger-size measurement or memory guarantee. Sampled memory is "
            "reported separately; multiple copies and GPU histogram memory "
            "must also fit. Positive final paired gains indicate continued "
            "learning, not a plateau."
        )
    }