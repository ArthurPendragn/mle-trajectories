import numpy as np
import pandas as pd
import skrub
import torch
from sklearn.base import BaseEstimator, RegressorMixin

def ride_features(X):
    coordinates = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    raw = X[coordinates + ['passenger_count']].replace([np.inf, -np.inf], np.nan)
    flags, clean = {}, {}
    for endpoint in ['pickup', 'dropoff']:
        lon, lat = raw[endpoint + '_longitude'], raw[endpoint + '_latitude']
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
        east_km=east, north_km=north, abs_east_km=east.abs(),
        abs_north_km=north.abs(),
        euclidean_km=(east ** 2 + north ** 2).skb.apply_func(np.sqrt),
        manhattan_km=east.abs() + north.abs()
    ).drop(columns=['passenger_count'])
    airport_features = {}
    for airport, lon, lat in [('jfk', -73.7781, 40.6413), ('lga', -73.874, 40.7769), ('ewr', -74.1745, 40.6895)]:
        for endpoint in ['pickup', 'dropoff']:
            dx = ((clean[endpoint + '_longitude'] - lon) * 84.3).clip(-300, 300)
            dy = ((clean[endpoint + '_latitude'] - lat) * 111.32).clip(-300, 300)
            airport_features[endpoint + '_' + airport + '_km'] = (dx ** 2 + dy ** 2).skb.apply_func(np.sqrt)
    distance = distance.assign(**airport_features).replace([np.inf, -np.inf], np.nan).astype('float32').skb.set_name('ride_geometry')
    timestamp = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True, errors='coerce')
    dates = X[['passenger_count']].assign(
        year=timestamp.dt.year, month=timestamp.dt.month, day=timestamp.dt.day,
        weekday=timestamp.dt.dayofweek, hour=timestamp.dt.hour,
        elapsed_days=(timestamp - pd.Timestamp('2009-01-01', tz='UTC')).dt.days
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('calendar')
    return location.skb.concat([distance, dates], axis=1)

class CudaSplineRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, epochs=6, batch_size=32768, learning_rate=0.03, l2=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2

    def _basis(self, values, count):
        centers = torch.arange(count, device=values.device, dtype=values.dtype)
        position = values.clamp(0, 1).unsqueeze(-1) * (count - 3) + 1
        a = (position - centers).abs()
        return torch.where(a < 1, 2 / 3 - a.square() + 0.5 * a.pow(3),
                           torch.clamp(2 - a, min=0).pow(3) / 6)

    def _design(self, x):
        valid = torch.isfinite(x)
        x = torch.where(valid, x, self.median_t_)
        z = ((x - self.median_t_) / self.scale_t_).clamp(-10, 10)
        u = (x[:, self.spline_indices_] - self.low_t_) / self.width_t_
        # Eight cubic basis functions keep the full interaction design below 512.
        b = self._basis(u, 8)
        blocks = [z, b.flatten(1)]
        small = self._basis(u[:, :4], 6)
        for left, right in [(0, 1), (2, 3), (0, 2), (1, 3)]:
            blocks.append((small[:, left, :, None] * small[:, right, None, :]).flatten(1))
        hour = x[:, self.hour_index_]
        phase = hour * (2 * np.pi / 24)
        cyclic = torch.stack([phase.sin(), phase.cos(), (2 * phase).sin(), (2 * phase).cos()], dim=1)
        year = x[:, self.year_index_].round().long()
        years = (year[:, None] == torch.arange(2009, 2016, device=x.device)[None, :]).float()
        month = x[:, self.month_index_].round().long()
        weekday = x[:, self.weekday_index_].round().long()
        blocks.extend([
            cyclic, years,
            (month[:, None] == torch.arange(1, 13, device=x.device)[None, :]).float(),
            (weekday[:, None] == torch.arange(7, device=x.device)[None, :]).float(),
            (b[:, 4, :, None] * years[:, None, :]).flatten(1),
            (b[:, 4, :, None] * cyclic[:, None, :]).flatten(1)
        ])
        return torch.cat(blocks, dim=1)

    def fit(self, X, y):
        torch.set_num_threads(8)
        torch.manual_seed(73191)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.columns_ = list(X.columns)
        a = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32)
        rng = np.random.RandomState(73191)
        sample = a[rng.choice(len(a), min(200000, len(a)), replace=False)]
        sample = np.where(np.isfinite(sample), sample, np.nan)
        median = np.nanmedian(sample, axis=0)
        median = np.nan_to_num(median)
        q = np.nanpercentile(sample, [25, 75], axis=0)
        scale = np.maximum(np.nan_to_num(q[1] - q[0], nan=1), 0.01)
        names = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                 'euclidean_km', 'manhattan_km']
        names += [endpoint + '_' + airport + '_km'
                  for airport in ['jfk', 'lga', 'ewr'] for endpoint in ['pickup', 'dropoff']]
        self.spline_indices_ = [self.columns_.index(name) for name in names]
        low, high = [], []
        for name, index in zip(names, self.spline_indices_):
            values = sample[:, index]
            if 'longitude' in name:
                values = values[(values > -74.3) & (values < -73.6)]
            elif 'latitude' in name:
                values = values[(values > 40.45) & (values < 41)]
            values = values[np.isfinite(values)]
            bounds = np.quantile(values, [0.005, 0.995]) if len(values) else [0, 1]
            low.append(bounds[0])
            high.append(max(bounds[1], bounds[0] + 0.001))
        self.median_t_ = torch.tensor(median, dtype=torch.float32, device=self.device_)
        self.scale_t_ = torch.tensor(scale, dtype=torch.float32, device=self.device_)
        self.low_t_ = torch.tensor(low, dtype=torch.float32, device=self.device_)
        self.width_t_ = torch.tensor(np.array(high) - low, dtype=torch.float32, device=self.device_)
        self.hour_index_ = self.columns_.index('hour')
        self.year_index_ = self.columns_.index('year')
        self.month_index_ = self.columns_.index('month')
        self.weekday_index_ = self.columns_.index('weekday')
        with torch.no_grad():
            dimension = self._design(torch.tensor(a[:2], device=self.device_)).shape[1]
        if dimension > 512:
            raise ValueError('Spline feature cap exceeded')
        self.net_ = torch.nn.Linear(dimension, 1).to(self.device_)
        torch.nn.init.zeros_(self.net_.weight)
        torch.nn.init.constant_(self.net_.bias, float(np.mean(target)))
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)
        starts = np.arange(0, len(a), self.batch_size)
        for epoch in range(self.epochs):
            total = 0.0
            for start in rng.permutation(starts):
                stop = min(start + self.batch_size, len(a))
                xb = torch.tensor(a[start:stop], device=self.device_)
                yb = torch.tensor(target[start:stop], device=self.device_)
                design = self._design(xb)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.net_(design).squeeze(1)
                mse = (prediction - yb).square().mean()
                loss = mse + self.l2 * self.net_.weight.square().sum()
                loss.backward()
                optimizer.step()
                total += float(mse.detach()) * (stop - start)
            print('spline epoch', epoch + 1, 'training MSE', total / len(a), flush=True)
        return self

    def predict(self, X):
        a = np.asarray(X, dtype=np.float32)
        output = []
        self.net_.eval()
        with torch.no_grad():
            for start in range(0, len(a), self.batch_size):
                xb = torch.tensor(a[start:start + self.batch_size], device=self.device_)
                output.append(self.net_(self._design(xb)).squeeze(1).cpu().numpy())
        return np.concatenate(output)

def build():
    setup = build_evaluation()
    features = ride_features(setup['X']).skb.set_name('spline_inputs')
    pred = features.skb.apply(CudaSplineRegressor(), y=setup['y']).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}