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
        started = time.monotonic()
        callback = ElapsedTimeLimit(self.seconds).fit()
        self.model_ = XGBRegressor(
            objective="reg:squarederror", tree_method="hist", device="cuda",
            n_estimators=1200, max_depth=10, learning_rate=0.07,
            max_bin=256, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, n_jobs=8, random_state=73191,
            callbacks=[callback],
        )
        self.model_.fit(X, y)
        self.fit_seconds_ = time.monotonic() - started
        self.completed_rounds_ = self.model_.get_booster().num_boosted_rounds()
        print("completed_rounds=", self.completed_rounds_,
              "fit_seconds=", self.fit_seconds_, flush=True)
        return self

    def predict(self, X):
        return self.model_.predict(X)


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
    euclidean = (east ** 2 + north ** 2).skb.apply_func(np.sqrt)
    distance = X[["passenger_count"]].assign(
        east_km=east, north_km=north,
        abs_east_km=east.abs(), abs_north_km=north.abs(),
        euclidean_km=euclidean,
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

    angle = np.deg2rad(29.0)
    s, c = float(np.sin(angle)), float(np.cos(angle))
    along = east * s + north * c
    across = east * c - north * s
    rotated_distance = along.abs() + across.abs()
    regime = (timestamp >= pd.Timestamp("2012-09-04", tz="UTC")).fillna(False).astype("float32")
    rush = (
        timestamp.dt.dayofweek.between(0, 4)
        & timestamp.dt.hour.between(16, 19)
    ).fillna(False).astype("float32")
    overnight = (
        (timestamp.dt.hour >= 20) | (timestamp.dt.hour < 6)
    ).fillna(False).astype("float32")
    route = {
        "street_along_km": along,
        "street_across_km": across,
        "street_manhattan_km": rotated_distance,
        "bearing_east": east / euclidean.clip(lower=0.001),
        "bearing_north": north / euclidean.clip(lower=0.001),
        "zero_distance": euclidean.eq(0).fillna(False).astype("float32"),
        "post_2012_tariff": regime,
        "distance_post_tariff": euclidean * regime,
        "street_distance_post_tariff": rotated_distance * regime,
        "weekday_rush": rush,
        "overnight": overnight,
        "distance_rush": euclidean * rush,
        "distance_overnight": euclidean * overnight,
    }
    for endpoint in ["pickup", "dropoff"]:
        x = ((clean[endpoint + "_longitude"] + 73.985) * 84.3).clip(-300, 300)
        y = ((clean[endpoint + "_latitude"] - 40.75) * 111.32).clip(-300, 300)
        route[endpoint + "_street_along"] = x * s + y * c
        route[endpoint + "_street_across"] = x * c - y * s
        for airport in ["jfk", "lga", "ewr"]:
            proximity = airport_features[endpoint + "_" + airport + "_km"]
            near = proximity.le(3).fillna(False).astype("float32")
            route[endpoint + "_" + airport + "_near"] = near
            route[endpoint + "_" + airport + "_near_post_tariff"] = near * regime
            route[endpoint + "_" + airport + "_distance_post_tariff"] = proximity * regime

    # Fixed access-point proxies; these do not assert an observed route.
    for name, lon, lat in [
        ("queensboro", -73.9635, 40.7590),
        ("queens_midtown", -73.9714, 40.7461),
        ("williamsburg", -73.9840, 40.7128),
        ("brooklyn_bridge", -74.0048, 40.7069),
        ("battery_tunnel", -74.0140, 40.7006),
        ("holland_tunnel", -74.0110, 40.7250),
        ("lincoln_tunnel", -73.9965, 40.7595),
        ("george_washington", -73.9430, 40.8490),
        ("triborough", -73.9270, 40.8010),
    ]:
        endpoint_distances = {}
        for endpoint in ["pickup", "dropoff"]:
            dx = ((clean[endpoint + "_longitude"] - lon) * 84.3).clip(-300, 300)
            dy = ((clean[endpoint + "_latitude"] - lat) * 111.32).clip(-300, 300)
            proxy = (dx ** 2 + dy ** 2).skb.apply_func(np.sqrt)
            endpoint_distances[endpoint] = proxy
            route[endpoint + "_" + name + "_access_km"] = proxy
        via = endpoint_distances["pickup"] + endpoint_distances["dropoff"]
        route[name + "_via_km"] = via
        route[name + "_detour_km"] = via - euclidean

    route_block = X[["passenger_count"]].assign(**route).drop(
        columns=["passenger_count"]
    ).replace([np.inf, -np.inf], np.nan).astype("float32").skb.set_name("route_tariff")
    return location.skb.concat([distance, dates, route_block], axis=1)


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