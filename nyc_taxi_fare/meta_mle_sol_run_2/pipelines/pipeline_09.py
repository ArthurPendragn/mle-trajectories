import numpy as np
import pandas as pd
import skrub
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


class FareNetwork(nn.Module):
    def __init__(self, width_in):
        super().__init__()
        self.input_layer = nn.Linear(width_in, 256)
        self.hidden_one = nn.Linear(256, 256)
        self.hidden_two = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, 1)
        self.activation = nn.SiLU()

    def forward(self, X):
        h = self.activation(self.input_layer(X))
        h = h + self.activation(self.hidden_one(h))
        h = h + self.activation(self.hidden_two(h))
        return self.output_layer(h).squeeze(-1)


class FourierFareRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, epochs=6, batch_size=8192, learning_rate=0.001, seed=1971):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        torch.set_num_threads(8)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        values = np.asarray(X, dtype=np.float32)
        missing = ~np.isfinite(values)
        values = np.where(missing, np.nan, values)
        self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        imputed = self.imputer_.fit_transform(values)
        self.scaler_ = RobustScaler(quantile_range=(10, 90))
        scaled = self.scaler_.fit_transform(imputed)
        encoded = np.concatenate(
            [np.clip(scaled, -10, 10), missing.astype(np.float32),
             (np.abs(scaled) > 10).astype(np.float32)], axis=1
        ).astype(np.float32)
        self.n_features_in_ = values.shape[1]
        features = torch.as_tensor(encoded, device=self.device_)
        targets = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device_)
        self.net_ = FareNetwork(encoded.shape[1]).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.learning_rate, weight_decay=0.0001
        )
        self.net_.train()
        for epoch in range(self.epochs):
            order = torch.randperm(len(features), device=self.device_)
            total = torch.zeros((), device=self.device_)
            rate = self.learning_rate * (0.2 + 0.8 * (1 + np.cos(np.pi * epoch / self.epochs)) / 2)
            for group in optimizer.param_groups:
                group['lr'] = rate
            for start in range(0, len(features), self.batch_size):
                indices = order[start:start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction = self.net_(features[indices])
                loss = torch.mean((prediction - targets[indices]) ** 2)
                loss.backward()
                optimizer.step()
                total += loss.detach() * len(indices)
            print('epoch', epoch + 1, 'training_mse', (total / len(features)).item())
        self.net_.eval()
        return self

    def predict(self, X):
        values = np.asarray(X, dtype=np.float32)
        missing = ~np.isfinite(values)
        imputed = self.imputer_.transform(np.where(missing, np.nan, values))
        scaled = self.scaler_.transform(imputed)
        encoded = np.concatenate(
            [np.clip(scaled, -10, 10), missing.astype(np.float32),
             (np.abs(scaled) > 10).astype(np.float32)], axis=1
        ).astype(np.float32)
        predictions = []
        self.net_.eval()
        with torch.no_grad():
            for start in range(0, len(encoded), self.batch_size):
                batch = torch.as_tensor(encoded[start:start + self.batch_size], device=self.device_)
                predictions.append(self.net_(batch).cpu().numpy())
        result = np.concatenate(predictions)
        print('finite_prediction_fraction', np.isfinite(result).mean())
        return result


def build():
    setup = build_evaluation()
    X, y = setup['X'], setup['y']
    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True, errors='coerce')
    calendar = X[['passenger_count']].assign(
        year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
        weekday=dt.dt.dayofweek, hour=dt.dt.hour, minute=dt.dt.minute
    ).astype('float32').skb.set_name('calendar')
    dx = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.4
    dy = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.2
    geography = X[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].assign(
        dx=dx, dy=dy, distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs()
    ).astype('float32').skb.set_name('geography')
    angle = np.deg2rad(29.0)
    c, s = np.cos(angle), np.sin(angle)
    px = (X['pickup_longitude'] + 74.0) * 84.4
    py = (X['pickup_latitude'] - 40.75) * 111.2
    qx = (X['dropoff_longitude'] + 74.0) * 84.4
    qy = (X['dropoff_latitude'] - 40.75) * 111.2
    route = X[['passenger_count']].assign(
        pickup_grid_x=px * c - py * s, pickup_grid_y=px * s + py * c,
        dropoff_grid_x=qx * c - qy * s, dropoff_grid_y=qx * s + qy * c,
        grid_manhattan=(dx * c - dy * s).abs() + (dx * s + dy * c).abs(),
        bearing=dy.skb.apply_func(np.arctan2, dx),
        midpoint_x=(px + qx) * 0.5, midpoint_y=(py + qy) * 0.5,
        post_tariff=(dt >= pd.Timestamp('2012-09-04', tz='UTC')).fillna(False),
        night=((dt.dt.hour >= 20) | (dt.dt.hour < 6)).fillna(False),
        weekday_rush=((dt.dt.dayofweek < 5) & (dt.dt.hour >= 16) & (dt.dt.hour < 20)).fillna(False)
    ).drop(columns=['passenger_count'])
    for airport, lon, lat in [('jfk', -73.7781, 40.6413), ('lga', -73.874, 40.7769), ('newark', -74.1745, 40.6895)]:
        for endpoint in ['pickup', 'dropoff']:
            ax = (X[endpoint + '_longitude'] - lon) * 84.4
            ay = (X[endpoint + '_latitude'] - lat) * 111.2
            route = route.assign(**{
                endpoint + '_' + airport + '_distance': (ax ** 2 + ay ** 2) ** 0.5
            })
    route = route.astype('float32').skb.set_name('route_regime')
    fourier = X[['passenger_count']]
    for name, coordinate in [('px', px), ('py', py), ('qx', qx), ('qy', qy)]:
        for index, wavelength in enumerate([0.25, 1.0, 4.0, 16.0]):
            phase = coordinate * (2 * np.pi / wavelength)
            fourier = fourier.assign(**{
                name + '_sin_' + str(index): phase.skb.apply_func(np.sin),
                name + '_cos_' + str(index): phase.skb.apply_func(np.cos)
            })
    fourier = fourier.drop(columns=['passenger_count']).astype('float32').skb.set_name('spatial_fourier')
    cyclic = X[['passenger_count']]
    for name, value, period in [
        ('hour', dt.dt.hour + dt.dt.minute / 60, 24),
        ('weekday', dt.dt.dayofweek, 7),
        ('month', dt.dt.month - 1, 12)
    ]:
        phase = value * (2 * np.pi / period)
        cyclic = cyclic.assign(**{
            name + '_sin': phase.skb.apply_func(np.sin),
            name + '_cos': phase.skb.apply_func(np.cos)
        })
    cyclic = cyclic.drop(columns=['passenger_count']).astype('float32').skb.set_name('cyclic_calendar')
    features = geography.skb.concat([calendar, route, fourier, cyclic], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).skb.set_name('finite_features')
    pred = features.skb.apply(FourierFareRegressor(), y=y).skb.set_name('model')
    result = {'pred': pred, 'scoring': setup['scoring']}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result