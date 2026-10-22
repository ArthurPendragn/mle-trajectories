import time
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback


class ElapsedTimeLimit(TrainingCallback, BaseEstimator):
    def __init__(self, seconds=240):
        self.seconds = seconds

    def fit(self, X=None, y=None):
        self.started_ = time.monotonic()
        self.rounds_ = 0
        return self

    def before_training(self, model):
        if not hasattr(self, "started_"):
            self.fit()
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.rounds_ = epoch + 1
        return time.monotonic() - self.started_ >= self.seconds


class TimedCapacityRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, seconds=240):
        self.seconds = seconds

    def fit(self, X, y):
        # Start before data conversion, not only before the boosting loop.
        # Reserve more of the attempt for graph evaluation and fold overhead.
        started = time.monotonic()
        callback = ElapsedTimeLimit(self.seconds).fit()
        self.model_ = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            n_estimators=1200,
            max_depth=10,
            learning_rate=0.07,
            max_bin=256,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            n_jobs=8,
            random_state=73191,
            callbacks=[callback],
        )
        self.model_.fit(X, y)
        self.fit_seconds_ = time.monotonic() - started
        self.completed_rounds_ = self.model_.get_booster().num_boosted_rounds()
        print(
            "completed_rounds=", self.completed_rounds_,
            "fit_seconds=", self.fit_seconds_,
            flush=True,
        )
        return self

    def predict(self, X):
        return self.model_.predict(X)


def ride_features(X):
    coordinates = [
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
    ]
    raw = X[coordinates + ["passenger_count"]].replace(
        [np.inf, -np.inf], np.nan
    )
    flags = {}
    clean = {}
    for endpoint in ["pickup", "dropoff"]:
        lon = raw[endpoint + "_longitude"]
        lat = raw[endpoint + "_latitude"]
        valid = (
            lon.between(-180, 180) & lat.between(-90, 90)
        ).fillna(False)
        local = (
            lon.between(-75, -72) & lat.between(40, 42)
        ).fillna(False)
        flags[endpoint + "_geographic_valid"] = valid.astype("float32")
        flags[endpoint + "_nyc_region"] = local.astype("float32")
        clean[endpoint + "_longitude"] = lon.where(valid)
        clean[endpoint + "_latitude"] = lat.where(valid)
        for axis in ["longitude", "latitude"]:
            col = endpoint + "_" + axis
            flags[col + "_missing"] = raw[col].isna().astype("float32")
    location = raw.assign(**flags).astype("float32").skb.set_name("locations")
    east = (
        (clean["dropoff_longitude"] - clean["pickup_longitude"]) * 84.3
    ).clip(-300, 300)
    north = (
        (clean["dropoff_latitude"] - clean["pickup_latitude"]) * 111.32
    ).clip(-300, 300)
    distance = X[["passenger_count"]].assign(
        east_km=east,
        north_km=north,
        abs_east_km=east.abs(),
        abs_north_km=north.abs(),
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
            dx = (
                (clean[endpoint + "_longitude"] - lon) * 84.3
            ).clip(-300, 300)
            dy = (
                (clean[endpoint + "_latitude"] - lat) * 111.32
            ).clip(-300, 300)
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
        year=timestamp.dt.year,
        month=timestamp.dt.month,
        day=timestamp.dt.day,
        weekday=timestamp.dt.dayofweek,
        hour=timestamp.dt.hour,
        elapsed_days=(
            timestamp - pd.Timestamp("2009-01-01", tz="UTC")
        ).dt.days,
    ).drop(columns=["passenger_count"]).astype("float32").skb.set_name("calendar")
    return location.skb.concat([distance, dates], axis=1)


def build():
    setup = build_evaluation()
    features = ride_features(setup["X"])
    pred = features.skb.apply(
        TimedCapacityRegressor(seconds=240), y=setup["y"]
    ).skb.set_name("model")
    return {
        "pred": pred,
        "scoring": setup["scoring"],
        "row_keys": setup["row_keys"],
    }