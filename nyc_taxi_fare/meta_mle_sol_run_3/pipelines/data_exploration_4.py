import time
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

TRAIN = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
SEED = 714
MAX_TRAIN = 10000000
VALID_SIZE = 20000
COLUMNS = [
    "fare_amount", "pickup_datetime", "pickup_longitude",
    "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "passenger_count",
]


class LearningCurve(TransformerMixin, BaseEstimator):
    def __init__(self, study_limit=1300):
        self.study_limit = study_limit

    def fit(self, X, y=None):
        started = time.perf_counter()
        records = []
        previous_scores = None
        previous_seconds = None
        previous_size = None
        for size in (300000, 1000000, 3000000, 10000000):
            projected = (
                previous_seconds * size / previous_size * 1.35
                if previous_seconds is not None else 25.0
            )
            elapsed = time.perf_counter() - started
            if records and elapsed + projected > self.study_limit:
                break
            experiment_started = time.perf_counter()
            scores = []
            fold_seconds = []
            for fold in range(3):
                fold_started = time.perf_counter()
                train = X.iloc[:size]
                valid = X.iloc[
                    MAX_TRAIN + fold * VALID_SIZE:
                    MAX_TRAIN + (fold + 1) * VALID_SIZE
                ]
                train_x = train.drop(columns="fare_amount")
                valid_x = valid.drop(columns="fare_amount")
                train_y = train["fare_amount"].to_numpy()
                valid_y = valid["fare_amount"].to_numpy()
                model = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    XGBRegressor(
                        objective="reg:squarederror",
                        device="cuda",
                        tree_method="hist",
                        n_estimators=700,
                        max_depth=8,
                        learning_rate=0.06,
                        min_child_weight=20,
                        reg_lambda=10,
                        random_state=SEED,
                        n_jobs=16,
                    ),
                )
                model.fit(train_x, train_y)
                prediction = model.predict(valid_x)
                scores.append(float(np.sqrt(np.mean((prediction - valid_y) ** 2))))
                fold_seconds.append(time.perf_counter() - fold_started)
                del model
            seconds = time.perf_counter() - experiment_started
            gains = (
                np.asarray(previous_scores) - np.asarray(scores)
                if previous_scores is not None else np.full(3, np.nan)
            )
            record = {
                "training_rows": size,
                "mean_rmse": float(np.mean(scores)),
                "experiment_seconds": seconds,
                "paired_gain": float(np.mean(gains)),
                "paired_gain_se": float(np.std(gains, ddof=1) / np.sqrt(3)),
            }
            for fold in range(3):
                record[f"rmse_fold_{fold}"] = scores[fold]
                record[f"seconds_fold_{fold}"] = fold_seconds[fold]
                record[f"gain_fold_{fold}"] = gains[fold]
            records.append(record)
            print(record, flush=True)
            previous_scores = scores
            previous_seconds = seconds
            previous_size = size
        self.results_ = pd.DataFrame(records)
        last = records[-1]
        available_fit_seconds = min(1800 - 180, (85691 / 12) - 180)
        projected_limit = int(
            last["training_rows"] * available_fit_seconds
            / (1.35 * last["experiment_seconds"])
        )
        projected_limit = min(55000000, projected_limit)
        uncertain = (
            np.isfinite(last["paired_gain_se"])
            and last["paired_gain"] <= 2 * last["paired_gain_se"]
        )
        self.results_["projected_affordable_rows"] = projected_limit
        self.results_["recommended_rows"] = (
            last["training_rows"] if uncertain else projected_limit
        )
        self.results_["final_gain_uncertain_not_proven_saturation"] = uncertain
        self.results_["study_seconds"] = time.perf_counter() - started
        return self

    def transform(self, X):
        return self.results_.copy()


def build():
    source = skrub.as_data_op(TRAIN).skb.apply_func(pd.read_csv, usecols=COLUMNS)
    rows = source.sample(n=MAX_TRAIN + 3 * VALID_SIZE, random_state=SEED).reset_index(drop=True)
    timestamp = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    features = rows.drop(columns="pickup_datetime")
    features = features.assign(
        year=timestamp.dt.year,
        month=timestamp.dt.month,
        day=timestamp.dt.day,
        weekday=timestamp.dt.dayofweek,
        hour=timestamp.dt.hour,
        minute=timestamp.dt.minute,
        second=timestamp.dt.second,
    )
    dx = (rows["dropoff_longitude"] - rows["pickup_longitude"]) * 84.3
    dy = (rows["dropoff_latitude"] - rows["pickup_latitude"]) * 111.2
    features = features.assign(
        delta_x=dx,
        delta_y=dy,
        distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan_distance=dx.abs() + dy.abs(),
        midpoint_longitude=(rows["pickup_longitude"] + rows["dropoff_longitude"]) / 2,
        midpoint_latitude=(rows["pickup_latitude"] + rows["dropoff_latitude"]) / 2,
    )
    results = features.skb.apply(LearningCurve())
    return {
        "curve": results,
        "design": skrub.as_data_op({
            "sampling": "Read complete source; sample 10060000 rows with seed 714; reset index. No exclusions. Nested training prefixes; validation slices start at 10000000, 10020000, 10040000.",
            "features": "Raw coordinates and passengers; UTC calendar and clock components; planar signed displacement, Euclidean and Manhattan distances, midpoint coordinates. No key.",
            "model": "Median imputation with missing indicators fitted separately per training pool; CUDA XGBoost squared error, 700 trees, depth 8, learning rate .06, child weight 20, lambda 10, seed 714.",
            "timing_scope": "Experiment times include slicing, array construction, fitted imputation, training and scoring all three folds. Shared recorded-source reading, sampling, timestamp and geographic graph operations are additionally measured by the harness wall timer, not included in experiment_seconds.",
            "limits": "Study fitting budget 1300 seconds; next size requires conservative 1.35x linear projection to fit. Affordability projection reserves 180 seconds for common graph overhead and requires twelve experiments within 85691 seconds and each below 1800 seconds.",
            "uncertainty": "A last paired gain at most twice its three-fold standard error is uncertain, not proof of saturation; recommendation then uses the largest measured size.",
        }),
    }