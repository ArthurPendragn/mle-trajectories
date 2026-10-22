import time
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback


class ElapsedTimeLimit(TrainingCallback, BaseEstimator):
    def __init__(self, seconds=240):
        self.seconds = seconds

    def fit(self, X=None, y=None):
        self.started_ = time.monotonic()
        return self

    def before_training(self, model):
        if not hasattr(self, "started_"):
            self.fit()
        return model

    def after_iteration(self, model, epoch, evals_log):
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
        print("completed_rounds=", self.model_.get_booster().num_boosted_rounds(),
              "fit_seconds=", time.monotonic() - started, flush=True)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class CrossFittedRouteEncoder(TransformerMixin, BaseEstimator):
    """Endpoint prior: 100 rides; directed route prior: 20 rides."""
    def __init__(self, resolutions=(0.0025, 0.0075)):
        self.resolutions = resolutions

    def _keys(self, X, resolution):
        regime = (X["elapsed_days"].to_numpy() >=
                  (pd.Timestamp("2012-09-04") - pd.Timestamp("2009-01-01")).days)
        cells = []
        for endpoint in ("pickup", "dropoff"):
            lon = X[endpoint + "_longitude"].to_numpy()
            lat = X[endpoint + "_latitude"].to_numpy()
            valid = (np.isfinite(lon) & np.isfinite(lat) &
                     (lon >= -75) & (lon <= -72) &
                     (lat >= 40) & (lat <= 42))
            ix = np.floor((np.where(valid, lon, -75) + 75) / resolution).astype(np.int64)
            iy = np.floor((np.where(valid, lat, 40) - 40) / resolution).astype(np.int64)
            cell = ix + 2048 * iy
            cells.append(np.where(valid, cell, -1))
        p, d = cells
        endpoint_span = 4194304
        pk = np.where(p >= 0, p + regime * endpoint_span, -1)
        dk = np.where(d >= 0, d + regime * endpoint_span, -1)
        route = np.where((p >= 0) & (d >= 0),
                         p * endpoint_span + d +
                         regime * endpoint_span * endpoint_span, -1)
        return pk, dk, route

    def _table(self, keys, y, mask):
        keep = mask & (keys >= 0)
        return pd.DataFrame({"key": keys[keep], "fare": y[keep]}).groupby(
            "key", sort=False
        )["fare"].agg(["sum", "count"])

    def _lookup(self, table, keys):
        selected = table.reindex(keys)
        sums = selected["sum"].fillna(0).to_numpy()
        counts = selected["count"].fillna(0).to_numpy()
        return sums, counts

    def _encode(self, keys, tables, prior):
        ps, pc = self._lookup(tables[0], keys[0])
        ds, dc = self._lookup(tables[1], keys[1])
        rs, rc = self._lookup(tables[2], keys[2])
        pm = (ps + 100 * prior) / (pc + 100)
        dm = (ds + 100 * prior) / (dc + 100)
        backoff = (pm + dm) / 2
        mean = (rs + 20 * backoff) / (rc + 20)
        return np.column_stack([
            mean, np.log1p(rc), pm, dm,
            np.log1p(pc), np.log1p(dc),
            (keys[2] < 0).astype(np.float32),
            (rc == 0).astype(np.float32),
        ]).astype(np.float32)

    def _attach(self, X, values):
        names = []
        for level in range(len(self.resolutions)):
            for suffix in ("fare", "log_support", "pickup_fare", "dropoff_fare",
                           "pickup_log_support", "dropoff_log_support",
                           "invalid", "unsupported"):
                names.append("local_%d_%s" % (level, suffix))
        return pd.concat([
            X, pd.DataFrame(values, index=X.index, columns=names)
        ], axis=1)

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        self.prior_ = float(np.mean(y))
        mask = np.ones(len(X), dtype=bool)
        self.tables_ = []
        for resolution in self.resolutions:
            keys = self._keys(X, resolution)
            self.tables_.append([self._table(k, y, mask) for k in keys])
        return self

    def fit_transform(self, X, y=None, **fit_params):
        started = time.monotonic()
        y = np.asarray(y, dtype=np.float64)
        hours = (X["elapsed_days"].fillna(-1).to_numpy(dtype=np.int64) * 24 +
                 X["hour"].fillna(0).to_numpy(dtype=np.int64))
        groups = ((hours * 1103515245 + 73191) % 2147483647) % 4
        values = np.empty((len(X), 8 * len(self.resolutions)), dtype=np.float32)
        self.tables_ = []
        self.prior_ = float(y.mean())
        all_rows = np.ones(len(X), dtype=bool)
        for level, resolution in enumerate(self.resolutions):
            keys = self._keys(X, resolution)
            for group in range(4):
                held = groups == group
                donor = ~held
                prior = float(y[donor].mean())
                tables = [self._table(k, y, donor) for k in keys]
                encoded = self._encode(tuple(k[held] for k in keys), tables, prior)
                values[held, level * 8:(level + 1) * 8] = encoded
                del tables, encoded
            self.tables_.append([self._table(k, y, all_rows) for k in keys])
            local = values[:, level * 8]
            covered = values[:, level * 8 + 7] == 0
            print("route_crossfit", resolution,
                  "coverage", float(covered.mean()),
                  "standalone_rmse", float(np.sqrt(np.mean((local - y) ** 2))),
                  "covered_rmse", float(np.sqrt(np.mean((local[covered] - y[covered]) ** 2))),
                  "fare_quantiles", np.quantile(local, [0.01, 0.5, 0.99]),
                  flush=True)
        print("route_encoding_seconds", time.monotonic() - started, flush=True)
        return self._attach(X, values)

    def transform(self, X):
        blocks = []
        for resolution, tables in zip(self.resolutions, self.tables_):
            encoded = self._encode(self._keys(X, resolution), tables, self.prior_)
            print("route_prediction_audit", resolution, "rows", len(X),
                  "coverage", float((encoded[:, 7] == 0).mean()),
                  "fare_quantiles", np.quantile(encoded[:, 0], [0.01, 0.5, 0.99]),
                  flush=True)
            blocks.append(encoded)
        return self._attach(X, np.column_stack(blocks))


def ride_features(X):
    coordinates = ["pickup_longitude", "pickup_latitude",
                   "dropoff_longitude", "dropoff_latitude"]
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
        manhattan_km=east.abs() + north.abs()
    ).drop(columns=["passenger_count"])
    airport_features = {}
    for airport, lon, lat in [
        ("jfk", -73.7781, 40.6413), ("lga", -73.874, 40.7769),
        ("ewr", -74.1745, 40.6895)
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
        elapsed_days=(timestamp - pd.Timestamp("2009-01-01", tz="UTC")).dt.days
    ).drop(columns=["passenger_count"]).astype("float32").skb.set_name("calendar")
    return location.skb.concat([distance, dates], axis=1)


def build():
    setup = build_evaluation()
    features = ride_features(setup["X"])
    features = features.skb.apply(
        CrossFittedRouteEncoder(), y=setup["y"]
    ).skb.set_name("cross_fitted_route_locality")
    pred = features.skb.apply(
        TimedCapacityRegressor(seconds=240), y=setup["y"]
    ).skb.set_name("model")
    return {
        "pred": pred,
        "scoring": setup["scoring"],
        "row_keys": setup["row_keys"],
    }