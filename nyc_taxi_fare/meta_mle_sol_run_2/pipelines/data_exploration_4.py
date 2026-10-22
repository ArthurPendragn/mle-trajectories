import time
import resource
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor

ROOT = "gs://mle-trajectories-data/nyc_taxi_fare/input"
COLS = ["fare_amount", "pickup_datetime", "pickup_longitude",
        "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
        "passenger_count"]


class TimerStart(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return {"X": X, "start": time.perf_counter()}

    def transform(self, X):
        return {"X": X, "start": time.perf_counter()}


class TimerStop(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return {"X": X["X"], "seconds": time.perf_counter() - X["start"]}

    def transform(self, X):
        return self.fit_transform(X)


class TimedCurvePoint(TransformerMixin, BaseEstimator):
    def __init__(self, n=100000):
        self.n = n

    def fit(self, X, y=None):
        features = X["features"]
        target = X["target"]
        n = self.n
        scores = []
        for fold in range(3):
            model = XGBRegressor(
                objective="reg:squarederror", device="cuda",
                tree_method="hist", n_estimators=600, max_depth=8,
                learning_rate=0.06, min_child_weight=20,
                subsample=0.85, colsample_bytree=0.9,
                reg_lambda=10, n_jobs=8, random_state=1971
            )
            model.fit(features.iloc[:n], target.iloc[:n])
            lo = n + fold * 20000
            hi = lo + 20000
            pred = model.predict(features.iloc[lo:hi])
            scores.append(float(np.sqrt(np.mean(
                (pred - target.iloc[lo:hi].to_numpy()) ** 2))))
            del model
        elapsed = time.perf_counter() - X["start"]
        row = {
            "training_rows": n,
            "validation_rows": 60000,
            "complete_three_fold_seconds": elapsed,
            "source_decode_seconds": X["decode_seconds"],
            "complete_with_decode_seconds": elapsed + X["decode_seconds"],
            "process_peak_host_gib": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
            "feature_frame_gib": float(
                features.memory_usage(deep=True).sum() / 1024 ** 3),
        }
        for i, score in enumerate(scores):
            row["rmse_fold_" + str(i)] = score
        self.report_ = pd.DataFrame([row])
        return self

    def transform(self, X):
        return self.report_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class BudgetRecommendation(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        a = X.sort_values("training_rows").reset_index(drop=True)
        rows = []
        for i in range(len(a)):
            row = a.iloc[i].to_dict()
            if i:
                gains = np.array([
                    a.loc[i - 1, "rmse_fold_" + str(j)]
                    - a.loc[i, "rmse_fold_" + str(j)]
                    for j in range(3)
                ])
                row["paired_gain_mean"] = float(gains.mean())
                row["paired_gain_se"] = float(gains.std(ddof=1) / np.sqrt(3))
                for j in range(3):
                    row["paired_gain_fold_" + str(j)] = gains[j]
            rows.append(row)
        self.curve_ = pd.DataFrame(rows)
        last = a.iloc[-1]
        previous = a.iloc[-2]
        slope = max(
            (last.complete_three_fold_seconds
             - previous.complete_three_fold_seconds)
            / (last.training_rows - previous.training_rows),
            last.complete_three_fold_seconds / last.training_rows
        )
        intercept = max(
            0.0, last.complete_three_fold_seconds
            - slope * last.training_rows
        )
        decode = float(last.source_decode_seconds)
        limit = min(1500.0, 26000.0 / 12.0)
        runtime_cap = int(max(
            1000000,
            ((limit - decode) / 3.0 - intercept) / slope
        ))
        host_base = float(a.process_peak_host_gib.max())
        host_cap = int(max(1000000, (100.0 - host_base) * 1024 ** 3 / 2000))
        gpu_cap = int(65.0 * 1024 ** 3 / 2000)
        chosen = min(runtime_cap, host_cap, gpu_cap, 55000000)
        chosen = max(1000000, chosen // 100000 * 100000)
        estimate = decode + 3.0 * (intercept + slope * chosen)
        self.budget_ = pd.DataFrame([{
            "recommended_training_rows_per_fold": chosen,
            "locked_pool_rows": chosen + 60000,
            "fixed_validation_rows": 60000,
            "estimated_complete_seconds_with_headroom": estimate,
            "estimated_twelve_experiment_seconds": 12 * estimate,
            "runtime_cap_rows": runtime_cap,
            "host_cap_rows": host_cap,
            "gpu_cap_rows": gpu_cap,
            "host_allowance_bytes_per_added_row": 2000,
            "gpu_allowance_bytes_per_row": 2000,
            "gpu_peak_measured": False,
            "memory_caveat": "GPU usage is an engineering allowance, not measured peak; host peak is cumulative process RSS.",
            "design": "Shared training pool, three disjoint 20000-row holdouts; exclude all holdout timestamps from training.",
            "extrapolation_caveat": "Unmeasured sizes are extrapolations; no larger size was fitted."
        }])
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return {"curve": self.curve_, "recommendation": self.budget_}

    def transform(self, X):
        return {"curve": self.curve_, "recommendation": self.budget_}


def features(rows):
    dt = rows["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True)
    f = rows[COLS[2:]].copy()
    f = f.assign(
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        dayofweek=dt.dt.dayofweek,
        hour=dt.dt.hour,
        minute=dt.dt.minute
    )
    dx = (rows["dropoff_longitude"] - rows["pickup_longitude"]) * 84.4
    dy = (rows["dropoff_latitude"] - rows["pickup_latitude"]) * 111.2
    f = f.assign(
        dx=dx, dy=dy,
        distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs()
    )
    return f.astype("float32")


def build():
    start = skrub.as_data_op(ROOT + "/train.csv.gz").skb.apply(
        TimerStart(), no_wrap=True)
    raw = start["X"].skb.apply_func(pd.read_csv, usecols=COLS)
    decoded = skrub.as_data_op({
        "X": raw, "start": start["start"]
    }).skb.apply(TimerStop(), no_wrap=True)
    sampled = decoded["X"].sample(n=1400000, random_state=1971).reset_index(drop=True)
    validation = sampled.iloc[-60000:]
    eligible = sampled.iloc[:-60000]
    eligible = eligible[
        ~eligible["pickup_datetime"].isin(validation["pickup_datetime"])
    ]
    reports = []
    for n in [100000, 300000, 1000000]:
        rows = eligible.iloc[:n].skb.concat([validation], axis=0).reset_index(drop=True)
        timed = rows.skb.apply(TimerStart(), no_wrap=True)
        r = timed["X"]
        payload = skrub.as_data_op({
            "features": features(r),
            "target": r["fare_amount"],
            "start": timed["start"],
            "decode_seconds": decoded["seconds"]
        })
        reports.append(payload.skb.apply(TimedCurvePoint(n)))
    table = reports[0].skb.concat(reports[1:], axis=0)
    summary = table.skb.apply(BudgetRecommendation(), no_wrap=True)
    return {
        "curve": summary["curve"],
        "recommendation": summary["recommendation"],
        "eligible_training_count": eligible.shape[0],
        "protocol": skrub.as_data_op({
            "sampling": "Uniform full-source pandas sample, 1400000 rows, seed 1971.",
            "features": "Raw coordinates and passenger count; UTC year, month, day, weekday, hour, minute; signed dx=(drop_lon-pick_lon)*84.4, dy=(drop_lat-pick_lat)*111.2, Euclidean distance and unrotated Manhattan distance; all float32.",
            "timing": "Per-size timer precedes all feature nodes and ends after three sequential fits and predictions. Full CSV decoding is timed separately and added to every budget estimate. Sampling and timestamp filtering are excluded from per-size timers; factor-three budget allowance covers overhead.",
            "population": "Unfiltered. Last 60000 random-sample rows are fixed validation; every validation timestamp excluded from the common training pool.",
            "memory": "Reported host peak is cumulative process high-water RSS, not a size-specific incremental peak. GPU peak is not instrumented; recommendation uses explicit conservative memory allowances."
        })
    }