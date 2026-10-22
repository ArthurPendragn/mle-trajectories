import time
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
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
        callback = ElapsedTimeLimit(self.seconds).fit()
        self.model_ = XGBRegressor(
            objective="reg:squarederror", tree_method="hist", device="cuda",
            n_estimators=1200, max_depth=10, learning_rate=0.07,
            max_bin=256, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, n_jobs=8, random_state=73191,
            callbacks=[callback])
        self.model_.fit(X, y)
        print("completed_rounds", self.model_.get_booster().num_boosted_rounds(),
              flush=True)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class LocalJourneyStatistics(TransformerMixin, BaseEstimator):
    def __init__(self, pair_smoothing=20.0, endpoint_smoothing=100.0):
        self.pair_smoothing = pair_smoothing
        self.endpoint_smoothing = endpoint_smoothing

    def _keys(self, X, scale):
        year = X["year"].to_numpy()
        month = X["month"].to_numpy()
        regime = ((year > 2012) | ((year == 2012) & (month >= 9))).astype("int64")
        valid = np.ones(len(X), dtype=bool)
        cells = []
        for endpoint in ["pickup", "dropoff"]:
            lon = X[endpoint + "_longitude"].to_numpy(dtype="float64")
            lat = X[endpoint + "_latitude"].to_numpy(dtype="float64")
            endpoint_valid = (
                np.isfinite(lon) & np.isfinite(lat) &
                (lon >= -75) & (lon <= -72) & (lat >= 40) & (lat <= 42))
            valid &= endpoint_valid
            # Bounded local cells admit collision-free integer packing.
            x = np.floor(
                (np.where(endpoint_valid, lon, -75) + 75) * 84.3 / scale
            ).astype("int64")
            z = np.floor(
                (np.where(endpoint_valid, lat, 40) - 40) * 111.32 / scale
            ).astype("int64")
            cells.append(x * 1024 + z)
        pickup, dropoff = cells
        keys = {
            "pickup": pickup * 2 + regime + 1,
            "dropoff": dropoff * 2 + regime + 1,
            "pair": (pickup * 1048576 + dropoff) * 2 + regime + 1}
        for key in keys.values():
            key[~valid] = 0
        return keys

    def _values(self, totals, counts, prior):
        result = {}
        means = []
        for endpoint in ["pickup", "dropoff"]:
            count = counts[endpoint]
            mean = (
                totals[endpoint] + self.endpoint_smoothing * prior
            ) / (count + self.endpoint_smoothing)
            means.append(mean)
            result[endpoint + "_mean"] = mean.astype("float32")
            result[endpoint + "_log_support"] = np.log1p(count).astype("float32")
        count = counts["pair"]
        fallback = (means[0] + means[1]) / 2
        result["pair_mean"] = (
            (totals["pair"] + self.pair_smoothing * fallback) /
            (count + self.pair_smoothing)).astype("float32")
        result["pair_log_support"] = np.log1p(count).astype("float32")
        return result

    def _learn(self, X, y, cross_fit):
        target = np.asarray(y, dtype="float64")
        self.prior_ = float(target.mean())
        self.tables_ = {}
        output = {}
        if cross_fit:
            hours = (
                X["elapsed_days"].to_numpy(dtype="int64") * 24 +
                X["hour"].to_numpy(dtype="int64"))
            partition = pd.util.hash_array(hours) % 3
            selections = [np.flatnonzero(partition == fold) for fold in range(3)]
            target_sum = target.sum()
        for scale in [0.25, 1.0]:
            keys = self._keys(X, scale)
            tables = {}
            codes = {}
            for name, key in keys.items():
                inverse, unique = pd.factorize(key, sort=False)
                count = np.bincount(inverse, minlength=len(unique))
                total = np.bincount(inverse, weights=target, minlength=len(unique))
                invalid = unique == 0
                count[invalid] = 0
                total[invalid] = 0
                tables[name] = (pd.Index(unique), total, count)
                if cross_fit:
                    codes[name] = inverse
            self.tables_[scale] = tables
            if not cross_fit:
                continue
            encoded = {}
            for selected in selections:
                totals = {}
                counts = {}
                for name, inverse in codes.items():
                    index, total, count = tables[name]
                    held_codes = inverse[selected]
                    held_count = np.bincount(held_codes, minlength=len(index))
                    held_sum = np.bincount(
                        held_codes, weights=target[selected], minlength=len(index))
                    remaining_count = count - held_count
                    remaining_sum = total - held_sum
                    invalid = index.to_numpy() == 0
                    remaining_count[invalid] = 0
                    remaining_sum[invalid] = 0
                    counts[name] = remaining_count[held_codes]
                    totals[name] = remaining_sum[held_codes]
                prior = float(
                    (target_sum - target[selected].sum()) /
                    (len(target) - len(selected)))
                values = self._values(totals, counts, prior)
                for name, value in values.items():
                    if name not in encoded:
                        encoded[name] = np.empty(len(X), dtype="float32")
                    encoded[name][selected] = value
            for name, value in encoded.items():
                output["local_" + str(scale).replace(".", "_") + "_" + name] = value
        if cross_fit:
            result = pd.DataFrame(output, index=X.index)
            self._audit(result, "cross_fitted_training", target)
            return result
        return self

    def fit(self, X, y):
        return self._learn(X, y, False)

    def fit_transform(self, X, y=None, **fit_params):
        return self._learn(X, y, True)

    def transform(self, X):
        output = {}
        for scale in [0.25, 1.0]:
            totals = {}
            counts = {}
            for name, key in self._keys(X, scale).items():
                index, total, count = self.tables_[scale][name]
                positions = index.get_indexer(key)
                found = (positions >= 0) & (key != 0)
                totals[name] = np.zeros(len(X), dtype="float64")
                counts[name] = np.zeros(len(X), dtype="int64")
                totals[name][found] = total[positions[found]]
                counts[name][found] = count[positions[found]]
            for name, value in self._values(totals, counts, self.prior_).items():
                output["local_" + str(scale).replace(".", "_") + "_" + name] = value
        result = pd.DataFrame(output, index=X.index)
        self._audit(result, "held_out_or_prediction")
        return result

    def _audit(self, result, name, target=None):
        # Quantiles are diagnostic only; avoid sorting sixteen million rows.
        stride = max(1, len(result) // 100000)
        print("local_statistics_audit", name, "rows", len(result),
              "quantile_sample_stride", stride,
              "quantiles", result.iloc[::stride].quantile(
                  [0, .5, .9, .99]).to_dict(), flush=True)
        for column in result.columns:
            if column.endswith("pair_log_support"):
                covered = result[column].to_numpy() > 0
                print(column, "fallback_fraction", float(1 - covered.mean()),
                      flush=True)
                if target is not None and covered.any():
                    mean_column = column.replace("log_support", "mean")
                    residual = result[mean_column].to_numpy()[covered] - target[covered]
                    print("cross_fitted_standalone_rmse",
                          float(np.sqrt(np.mean(residual ** 2))), flush=True)


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
        east_km=east, north_km=north, abs_east_km=east.abs(), abs_north_km=north.abs(),
        euclidean_km=(east ** 2 + north ** 2).skb.apply_func(np.sqrt),
        manhattan_km=east.abs() + north.abs()).drop(columns=["passenger_count"])
    airport_features = {}
    for airport, lon, lat in [
        ("jfk", -73.7781, 40.6413), ("lga", -73.874, 40.7769),
        ("ewr", -74.1745, 40.6895)]:
        for endpoint in ["pickup", "dropoff"]:
            dx = ((clean[endpoint + "_longitude"] - lon) * 84.3).clip(-300, 300)
            dy = ((clean[endpoint + "_latitude"] - lat) * 111.32).clip(-300, 300)
            airport_features[endpoint + "_" + airport + "_km"] = (
                dx ** 2 + dy ** 2).skb.apply_func(np.sqrt)
    distance = distance.assign(**airport_features).replace(
        [np.inf, -np.inf], np.nan).astype("float32").skb.set_name("ride_geometry")
    timestamp = X["pickup_datetime"].skb.apply_func(
        pd.to_datetime, utc=True, errors="coerce")
    dates = X[["passenger_count"]].assign(
        year=timestamp.dt.year, month=timestamp.dt.month, day=timestamp.dt.day,
        weekday=timestamp.dt.dayofweek, hour=timestamp.dt.hour,
        elapsed_days=(timestamp - pd.Timestamp("2009-01-01", tz="UTC")).dt.days
    ).drop(columns=["passenger_count"]).astype("float32").skb.set_name("calendar")
    return location.skb.concat([distance, dates], axis=1)


def build():
    setup = build_evaluation()
    features = ride_features(setup["X"])
    local = features.skb.apply(
        LocalJourneyStatistics(), y=setup["y"]).skb.set_name("local_journey_statistics")
    combined = features.skb.concat([local], axis=1)
    pred = combined.skb.apply(
        TimedCapacityRegressor(seconds=240), y=setup["y"]).skb.set_name("model")
    return {"pred": pred, "scoring": setup["scoring"], "row_keys": setup["row_keys"]}