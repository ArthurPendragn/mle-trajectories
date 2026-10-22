import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def ride_features(X):
    coordinates = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
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
    cycles = {}
    for name, period in [('hour', 24), ('weekday', 7), ('month', 12)]:
        angle = dates[name] * (2 * np.pi / period)
        cycles[name + '_sin'] = angle.skb.apply_func(np.sin)
        cycles[name + '_cos'] = angle.skb.apply_func(np.cos)
    cyclic = X[['passenger_count']].assign(**cycles).drop(columns=['passenger_count']).astype('float32').skb.set_name('cyclic_calendar')
    return location.skb.concat([distance, dates, cyclic], axis=1)


class FareNetwork(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        return self.layers(X).squeeze(-1)


class DeviceBatchDataset(torch.utils.data.Dataset):
    def __init__(self, values, target, batch_size):
        self.values = values
        self.target = target
        self.batch_size = batch_size
        self.order = None

    def __len__(self):
        return (len(self.values) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        indices = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        return self.values[indices], self.target[indices]


class NeuralFareRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, epochs=4, batch_size=8192, learning_rate=0.001, random_state=73191):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        torch.set_num_threads(8)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.imputer_ = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        self.scaler_ = StandardScaler()
        values = self.imputer_.fit_transform(np.asarray(X, dtype=np.float32))
        values = self.scaler_.fit_transform(values).astype(np.float32, copy=False)
        target = np.asarray(y, dtype=np.float64).reshape(-1)
        self.target_mean_ = float(target.mean())
        self.target_scale_ = max(float(target.std()), 1e-6)
        target = ((target - self.target_mean_) / self.target_scale_).astype(np.float32)
        self.n_features_in_ = np.asarray(X).shape[1]
        n_features = values.shape[1]
        # Keep the training tensors on-device and gather whole batches there,
        # avoiding millions of Python per-row fetches and CPU collation.
        dataset = DeviceBatchDataset(
            torch.as_tensor(np.ascontiguousarray(values), device=self.device_),
            torch.as_tensor(target, device=self.device_),
            self.batch_size
        )
        del values, target
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, shuffle=False, num_workers=0
        )
        generator = torch.Generator(device=self.device_).manual_seed(self.random_state)
        self.net_ = FareNetwork(n_features).to(self.device_)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.learning_rate)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs * len(loader)
        )
        for epoch in range(self.epochs):
            dataset.order = torch.randperm(
                len(dataset.values), generator=generator, device=self.device_
            )
            self.net_.train()
            total = torch.zeros((), device=self.device_)
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = self.net_(xb)
                loss = nn.functional.mse_loss(prediction, yb)
                loss.backward()
                optimizer.step()
                schedule.step()
                total += loss.detach() * len(xb)
            print(
                'epoch', epoch + 1, 'training_mse_usd',
                float(total.cpu()) / len(dataset.values) * self.target_scale_ ** 2,
                flush=True
            )
        return self

    def predict(self, X):
        values = self.imputer_.transform(np.asarray(X, dtype=np.float32))
        values = self.scaler_.transform(values).astype(np.float32, copy=False)
        self.net_.eval()
        results = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                xb = torch.as_tensor(
                    np.ascontiguousarray(values[start:start + self.batch_size]),
                    device=self.device_
                )
                results.append(self.net_(xb).cpu().numpy())
        return np.concatenate(results).astype(np.float64) * self.target_scale_ + self.target_mean_


def build():
    setup = build_evaluation()
    features = ride_features(setup['X'])
    pred = features.skb.apply(NeuralFareRegressor(), y=setup['y']).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}