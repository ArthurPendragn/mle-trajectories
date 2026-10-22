import time
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback
from catboost import CatBoostRegressor


class ElapsedTimeLimit(TrainingCallback, BaseEstimator):
    def __init__(self, seconds=240):
        self.seconds = seconds

    def fit(self, X=None, y=None):
        self.started_ = time.monotonic()
        return self

    def before_training(self, model):
        self.fit()
        return model

    def after_iteration(self, model, epoch, evals_log):
        return time.monotonic() - self.started_ >= self.seconds


class InternalHourHoldout(TransformerMixin, BaseEstimator):
    def __init__(self, validation_size=30000, random_state=73191):
        self.validation_size = validation_size
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        groups = (
            X["elapsed_days"].to_numpy(dtype=np.float64) * 24
            + X["hour"].to_numpy(dtype=np.float64)
        )
        groups = np.nan_to_num(groups, nan=-999999)
        unique, inverse, counts = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        order = np.random.RandomState(self.random_state).permutation(len(unique))
        width = np.searchsorted(
            np.cumsum(counts[order]), self.validation_size
        ) + 1
        selected = np.zeros(len(unique), dtype=bool)
        selected[order[:width]] = True
        validation = np.flatnonzero(selected[inverse])
        training = np.flatnonzero(~selected[inverse])
        return {
            "X": X,
            "y": y,
            "X_val": X.iloc[validation],
            "y_val": np.asarray(y)[validation],
            "pilot_indices": training,
        }

    def transform(self, X):
        return {
            "X": X,
            "y": None,
            "X_val": None,
            "y_val": None,
            "pilot_indices": None,
        }


class ComplementaryEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, random_state=73191):
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None, pilot_indices=None):
        started = time.monotonic()
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(
            pilot_indices, size=min(800000, len(pilot_indices)), replace=False
        )
        pilot_X = X.iloc[indices]
        pilot_y = np.asarray(y)[indices]

        pilot_start = time.monotonic()
        pilot_xgb = XGBRegressor(
            objective="reg:squarederror", tree_method="hist", device="cuda",
            n_estimators=1200, max_depth=10, learning_rate=0.07,
            max_bin=256, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, n_jobs=8, random_state=self.random_state,
            callbacks=[ElapsedTimeLimit(15)],
        )
        pilot_xgb.fit(pilot_X, pilot_y)
        xgb_pilot_seconds = time.monotonic() - pilot_start

        # GPU CatBoost does not support Python stopping callbacks. Measure a
        # short pilot and conservatively extrapolate its full-pool iteration
        # count. This is a runtime estimate, not a hard wall-clock guarantee.
        cat_start = time.monotonic()
        pilot_cat = CatBoostRegressor(
            loss_function="RMSE", task_type="GPU", devices="0",
            iterations=40, depth=6, learning_rate=0.08,
            random_seed=self.random_state, thread_count=8,
            verbose=False, allow_writing_files=False,
        )
        pilot_cat.fit(pilot_X.fillna(-999), pilot_y)
        cat_pilot_seconds = time.monotonic() - cat_start

        a = pilot_xgb.predict(X_val).astype(np.float64)
        b = pilot_cat.predict(X_val.fillna(-999)).astype(np.float64)
        delta = b - a
        denominator = np.dot(delta, delta)
        self.cat_weight_ = float(
            np.clip(np.dot(np.asarray(y_val) - a, delta) / denominator, 0, 1)
        ) if denominator > 0 else 0.0

        self.pilot_rmse_ = {
            "xgb": float(np.sqrt(np.mean((a - y_val) ** 2))),
            "cat": float(np.sqrt(np.mean((b - y_val) ** 2))),
            "ensemble": float(np.sqrt(np.mean(
                (a + self.cat_weight_ * delta - y_val) ** 2
            ))),
        }
        del pilot_xgb, pilot_cat, pilot_X, pilot_y

        self.xgb_ = XGBRegressor(
            objective="reg:squarederror", tree_method="hist", device="cuda",
            n_estimators=1200, max_depth=10, learning_rate=0.07,
            max_bin=256, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, n_jobs=8, random_state=self.random_state,
            callbacks=[ElapsedTimeLimit(240)],
        )
        full_start = time.monotonic()
        self.xgb_.fit(X, y)
        xgb_seconds = time.monotonic() - full_start

        row_ratio = len(X) / len(indices)
        estimated_iteration_seconds = max(cat_pilot_seconds / 40, 0.001) * row_ratio
        iterations = max(1, min(400, int(60 / (1.25 * estimated_iteration_seconds))))
        self.cat_ = CatBoostRegressor(
            loss_function="RMSE", task_type="GPU", devices="0",
            iterations=iterations, depth=6, learning_rate=0.08,
            random_seed=self.random_state, thread_count=8,
            verbose=False, allow_writing_files=False,
        )
        full_start = time.monotonic()
        self.cat_.fit(X.fillna(-999), y)
        cat_seconds = time.monotonic() - full_start
        print({
            "cat_weight": self.cat_weight_,
            "internal_pilot_rmse": self.pilot_rmse_,
            "pilot_xgb_seconds": xgb_pilot_seconds,
            "pilot_cat_seconds": cat_pilot_seconds,
            "xgb_rounds": self.xgb_.get_booster().num_boosted_rounds(),
            "cat_rounds": self.cat_.tree_count_,
            "full_xgb_seconds": xgb_seconds,
            "full_cat_seconds": cat_seconds,
            "total_model_seconds": time.monotonic() - started,
            "limitations": (
                "Pilot weights transfer to full-pool models; CatBoost runtime "
                "uses a conservative throughput estimate, not a hard cap. "
                "Only ensemble outer scores are available to this estimator."
            ),
        }, flush=True)
        return self

    def predict(self, X):
        a = self.xgb_.predict(X)
        b = self.cat_.predict(X.fillna(-999))
        return (1 - self.cat_weight_) * a + self.cat_weight_ * b


def ride_features(X):
    coordinates = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
    ]
    raw = X[coordinates + ["passenger_count"]].replace([np.inf, -np.inf], np.nan)
    flags = {}
    clean = {}
    for endpoint in ["pickup", "dropoff"]:
        lon = raw[endpoint + "_longitude"]
        lat = raw[endpoint + "_latitude"]
        valid = (lon.between(-180, 180) & lat.between(-90, 90)).fillna(False)
        local = (lon.between(-75, -72) & lat.between(40, 42)).fillna(False)
        flags[endpoint + "_geographic_valid"] = valid.astype("float32")
        flags[endpoint + "_nyc_region"] = local.astype("float32")
        clean[endpoint + "_longitude"] = lon.where(valid)
        clean[endpoint + "_latitude"] = lat.where(valid)
        for axis in ["longitude", "latitude"]:
            col = endpoint + "_" + axis
            flags[col + "_missing"] = raw[col].isna().astype("float32")
    location = raw.assign(**flags).astype("float32").skb.set_name("locations")
    east = ((clean["dropoff_longitude"] - clean["pickup_longitude"]) * 84.3).clip(-300, 300)
    north = ((clean["dropoff_latitude"] - clean["pickup_latitude"]) * 111.32).clip(-300, 300)
    distance = X[["passenger_count"]].assign(
        east_km=east, north_km=north,
        abs_east_km=east.abs(), abs_north_km=north.abs(),
        euclidean_km=(east ** 2 + north ** 2).skb.apply_func(np.sqrt),
        manhattan_km=east.abs() + north.abs(),
    ).drop(columns=["passenger_count"])
    airport_features = {}
    for airport, lon, lat in [
        ("jfk", -73.7781, 40.6413),
        ("lga", -73.874, 40.7769),
        ("ewr", -74.1745, 40.6895),
    ]:
        for endpoint in ["pickup", "dropoff"]:
            dx = ((clean[endpoint + "_longitude"] - lon) * 84.3).clip(-300, 300)
            dy = ((clean[endpoint + "_latitude"] - lat) * 111.32).clip(-300, 300)
            airport_features[endpoint + "_" + airport + "_km"] = (
                dx ** 2 + dy ** 2
            ).skb.apply_func(np.sqrt)
    distance = distance.assign(**airport_features).replace(
        [np.inf, -np.inf], np.nan
    ).astype("float32").skb.set_name("ride_geometry")
    timestamp = X["pickup_datetime"].skb.apply_func(
        pd.to_datetime, utc=True, errors="coerce"
    )
    dates = X[["passenger_count"]].assign(
        year=timestamp.dt.year, month=timestamp.dt.month,
        day=timestamp.dt.day, weekday=timestamp.dt.dayofweek,
        hour=timestamp.dt.hour,
        elapsed_days=(timestamp - pd.Timestamp("2009-01-01", tz="UTC")).dt.days,
    ).drop(columns=["passenger_count"]).astype("float32").skb.set_name("calendar")
    return location.skb.concat([distance, dates], axis=1)


def build():
    setup = build_evaluation()
    features = ride_features(setup["X"])
    parts = features.skb.apply(
        InternalHourHoldout(), y=setup["y"], no_wrap=True
    )
    pred = parts["X"].skb.apply(
        ComplementaryEnsemble(),
        y=parts["y"],
        fit_kwargs={
            "X_val": parts["X_val"],
            "y_val": parts["y_val"],
            "pilot_indices": parts["pilot_indices"],
        },
    ).skb.set_name("model")
    return {
        "pred": pred,
        "scoring": setup["scoring"],
        "row_keys": setup["row_keys"],
    }