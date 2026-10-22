import time
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback


class SpatialFareStatistics(TransformerMixin, BaseEstimator):
    def __init__(self, smoothing=50):
        self.smoothing = smoothing

    def _keys(self, X):
        result = []
        for resolution in [0.01, 0.04]:
            cells = []
            width = int(np.ceil(2 / resolution)) + 1
            for endpoint in ['pickup', 'dropoff']:
                lon = X[endpoint + '_longitude'].to_numpy(dtype=np.float64)
                lat = X[endpoint + '_latitude'].to_numpy(dtype=np.float64)
                valid = (
                    np.isfinite(lon) & np.isfinite(lat)
                    & (lon >= -75) & (lon < -72)
                    & (lat >= 40) & (lat < 42)
                )
                cell = np.full(len(X), -1, dtype=np.int64)
                cell[valid] = (
                    np.floor((lon[valid] + 75) / resolution).astype(np.int64) * width
                    + np.floor((lat[valid] - 40) / resolution).astype(np.int64)
                )
                cells.append(cell)
                if resolution == 0.01:
                    result.append((endpoint + '_cell', cell))
            capacity = (int(np.ceil(3 / resolution)) + 1) * width
            pair = cells[0] * capacity + cells[1]
            pair[(cells[0] < 0) | (cells[1] < 0)] = -1
            result.append(('route_' + str(resolution).replace('.', '_'), pair))
        return result

    def fit(self, X, y):
        target = np.asarray(y, dtype=np.float64)
        self.mean_ = float(target.mean())
        self.sum_ = float(target.sum())
        self.count_ = len(target)
        self.tables_ = {}
        for name, key in self._keys(X):
            valid = key >= 0
            frame = pd.DataFrame({'key': key[valid], 'target': target[valid]})
            self.tables_[name] = frame.groupby('key', sort=False)['target'].agg(['sum', 'count'])
        return self

    def _output(self, key, table, mean, subtract=None):
        values = table.reindex(key)
        sums = values['sum'].fillna(0).to_numpy()
        counts = values['count'].fillna(0).to_numpy(dtype=np.float64)
        if subtract is not None:
            sums = sums - subtract[0]
            counts = np.maximum(counts - subtract[1], 0)
        means = (sums + self.smoothing * mean) / (counts + self.smoothing)
        return np.log1p(counts).astype('float32'), means.astype('float32')

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        target = np.asarray(y, dtype=np.float64)
        hours = X['pickup_datetime'].astype(str).str.slice(0, 13)
        partition = (
            pd.util.hash_pandas_object(hours, index=False).to_numpy() % 5
        ).astype(np.int64)
        fold_count = np.bincount(partition, minlength=5)
        fold_sum = np.bincount(partition, weights=target, minlength=5)
        reference_mean = (self.sum_ - fold_sum[partition]) / (
            self.count_ - fold_count[partition]
        )
        output = {}
        for name, key in self._keys(X):
            valid = key >= 0
            composite = key * 5 + partition
            frame = pd.DataFrame({
                'key': composite[valid],
                'target': target[valid]
            })
            held = frame.groupby('key', sort=False)['target'].agg(['sum', 'count'])
            own = held.reindex(composite)
            logs, means = self._output(
                key, self.tables_[name], reference_mean,
                subtract=(
                    own['sum'].fillna(0).to_numpy(),
                    own['count'].fillna(0).to_numpy(dtype=np.float64)
                )
            )
            output[name + '_log_count'] = logs
            output[name + '_fare_mean'] = means
            print(
                'spatial_crossfit', name,
                'coverage', float(np.mean(logs > 0)),
                'mean_quantiles', np.quantile(means, [0, .5, .9, .99, 1]),
                'standalone_rmse', float(np.sqrt(np.mean((means - target) ** 2))),
                flush=True
            )
        return pd.DataFrame(output, index=X.index)

    def transform(self, X):
        output = {}
        for name, key in self._keys(X):
            logs, means = self._output(key, self.tables_[name], self.mean_)
            output[name + '_log_count'] = logs
            output[name + '_fare_mean'] = means
            print(
                'spatial_inference', name,
                'coverage', float(np.mean(logs > 0)),
                'mean_quantiles', np.quantile(means, [0, .5, .9, .99, 1]),
                flush=True
            )
        return pd.DataFrame(output, index=X.index)


class ElapsedTimeLimit(TrainingCallback, BaseEstimator):
    def __init__(self, seconds=240):
        self.seconds = seconds

    def fit(self, X=None, y=None):
        self.started_ = time.monotonic()
        self.rounds_ = 0
        return self

    def before_training(self, model):
        if not hasattr(self, 'started_'):
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
            objective='reg:squarederror', tree_method='hist', device='cuda',
            n_estimators=1200, max_depth=10, learning_rate=0.07,
            max_bin=256, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=1.0, n_jobs=8, random_state=73191,
            callbacks=[callback]
        )
        self.model_.fit(X, y)
        self.fit_seconds_ = time.monotonic() - started
        self.completed_rounds_ = self.model_.get_booster().num_boosted_rounds()
        print('completed_rounds=', self.completed_rounds_,
              'fit_seconds=', self.fit_seconds_, flush=True)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def ride_features(X):
    coordinates = [
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]
    raw = X[coordinates + ['passenger_count']].replace([np.inf, -np.inf], np.nan)
    flags = {}
    clean = {}
    for endpoint in ['pickup', 'dropoff']:
        lon = raw[endpoint + '_longitude']
        lat = raw[endpoint + '_latitude']
        valid = (lon.between(-180, 180) & lat.between(-90, 90)).fillna(False)
        local = (lon.between(-75, -72) & lat.between(40, 42)).fillna(False)
        flags[endpoint + '_geographic_valid'] = valid.astype('float32')
        flags[endpoint + '_nyc_region'] = local.astype('float32')
        clean[endpoint + '_longitude'] = lon.where(valid)
        clean[endpoint + '_latitude'] = lat.where(valid)
        for axis in ['longitude', 'latitude']:
            col = endpoint + '_' + axis
            flags[col + '_missing'] = raw[col].isna().astype('float32')
    location = raw.assign(**flags).astype('float32').skb.set_name('locations')
    east = ((clean['dropoff_longitude'] - clean['pickup_longitude']) * 84.3).clip(-300, 300)
    north = ((clean['dropoff_latitude'] - clean['pickup_latitude']) * 111.32).clip(-300, 300)
    distance = X[['passenger_count']].assign(
        east_km=east, north_km=north,
        abs_east_km=east.abs(), abs_north_km=north.abs(),
        euclidean_km=(east ** 2 + north ** 2).skb.apply_func(np.sqrt),
        manhattan_km=east.abs() + north.abs()
    ).drop(columns=['passenger_count'])
    airport_features = {}
    for airport, lon, lat in [
        ('jfk', -73.7781, 40.6413),
        ('lga', -73.874, 40.7769),
        ('ewr', -74.1745, 40.6895)
    ]:
        for endpoint in ['pickup', 'dropoff']:
            dx = ((clean[endpoint + '_longitude'] - lon) * 84.3).clip(-300, 300)
            dy = ((clean[endpoint + '_latitude'] - lat) * 111.32).clip(-300, 300)
            airport_features[endpoint + '_' + airport + '_km'] = (
                dx ** 2 + dy ** 2
            ).skb.apply_func(np.sqrt)
    distance = distance.assign(**airport_features).replace(
        [np.inf, -np.inf], np.nan
    ).astype('float32').skb.set_name('ride_geometry')
    timestamp = X['pickup_datetime'].skb.apply_func(
        pd.to_datetime, utc=True, errors='coerce'
    )
    dates = X[['passenger_count']].assign(
        year=timestamp.dt.year, month=timestamp.dt.month,
        day=timestamp.dt.day, weekday=timestamp.dt.dayofweek,
        hour=timestamp.dt.hour,
        elapsed_days=(timestamp - pd.Timestamp('2009-01-01', tz='UTC')).dt.days
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('calendar')
    return location.skb.concat([distance, dates], axis=1)


def build():
    setup = build_evaluation()
    X, y = setup['X'], setup['y']
    spatial = X.skb.apply(
        SpatialFareStatistics(smoothing=50), y=y
    ).skb.set_name('spatial_statistics')
    features = ride_features(X).skb.concat([spatial], axis=1)
    pred = features.skb.apply(
        TimedCapacityRegressor(seconds=240), y=y
    ).skb.set_name('model')
    return {
        'pred': pred,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys']
    }