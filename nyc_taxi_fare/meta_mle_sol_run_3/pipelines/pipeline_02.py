import time
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class FareNetwork(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, width),
            nn.SiLU(),
            ResidualBlock(width),
            ResidualBlock(width),
            ResidualBlock(width),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class NeuralFareRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, width=256, epochs=8, batch_size=16384,
                 learning_rate=0.001, weight_decay=0.0001, seed=714):
        self.width = width
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed

    def fit(self, X, y):
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(16)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.imputer_ = SimpleImputer(strategy='median', add_indicator=True)
        self.scaler_ = StandardScaler()
        values = np.asarray(X, dtype=np.float32)
        values = self.imputer_.fit_transform(values)
        values = self.scaler_.fit_transform(values)
        np.clip(values, -10, 10, out=values)
        values = np.ascontiguousarray(values, dtype=np.float32)
        targets = np.asarray(y, dtype=np.float32).reshape(-1)
        self.n_features_in_ = X.shape[1]
        self.net_ = FareNetwork(values.shape[1], self.width).to(self.device_)
        features = torch.from_numpy(values).to(self.device_)
        labels = torch.from_numpy(targets.copy()).to(self.device_)
        del values, targets
        if self.device_ == 'cuda':
            torch.cuda.synchronize()
        print('preprocessing_seconds', time.perf_counter() - started, flush=True)
        started = time.perf_counter()
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        batches = (len(labels) + self.batch_size - 1) // self.batch_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs * batches,
        )
        generator = torch.Generator(device=self.device_)
        generator.manual_seed(self.seed)
        self.net_.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(
                len(labels), generator=generator, device=self.device_,
            )
            loss_sum = torch.zeros((), device=self.device_)
            for start in range(0, len(labels), self.batch_size):
                indices = permutation[start:start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction = self.net_(features[indices])
                loss = nn.functional.mse_loss(prediction, labels[indices])
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_sum += loss.detach() * len(indices)
            print('epoch', epoch + 1, 'training_mse',
                  (loss_sum / len(labels)).item(), flush=True)
        if self.device_ == 'cuda':
            torch.cuda.synchronize()
        print('training_seconds', time.perf_counter() - started, flush=True)
        del features, labels
        self.net_.eval()
        return self

    def predict(self, X):
        started = time.perf_counter()
        result = np.empty(len(X), dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                stop = min(start + self.batch_size, len(X))
                batch = np.asarray(X.iloc[start:stop], dtype=np.float32)
                batch = self.imputer_.transform(batch)
                batch = self.scaler_.transform(batch)
                np.clip(batch, -10, 10, out=batch)
                tensor = torch.from_numpy(
                    np.ascontiguousarray(batch, dtype=np.float32)
                ).to(self.device_)
                result[start:stop] = self.net_(tensor).cpu().numpy()
        print('prediction_seconds', time.perf_counter() - started, flush=True)
        return result


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    coordinates = X[
        ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
         'dropoff_latitude', 'passenger_count']
    ].astype('float32').skb.set_name('raw_coordinates')
    timestamp = X['pickup_datetime'].skb.apply_func(
        pd.to_datetime, utc=True, errors='coerce'
    )
    dates = coordinates[['passenger_count']].assign(
        year=timestamp.dt.year,
        month=timestamp.dt.month,
        day=timestamp.dt.day,
        weekday=timestamp.dt.dayofweek,
        hour=timestamp.dt.hour,
        minute=timestamp.dt.minute,
        second=timestamp.dt.second,
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('dates')
    dx = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.3
    dy = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.2
    geometry = coordinates[['passenger_count']].assign(
        delta_x=dx,
        delta_y=dy,
        absolute_x=dx.abs(),
        absolute_y=dy.abs(),
        euclidean_distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan_distance=dx.abs() + dy.abs(),
        midpoint_longitude=(
            X['pickup_longitude'] + X['dropoff_longitude']
        ) / 2,
        midpoint_latitude=(
            X['pickup_latitude'] + X['dropoff_latitude']
        ) / 2,
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('geometry')
    features = coordinates.skb.concat([dates, geometry], axis=1)
    features = features.replace(
        [float('inf'), -float('inf')], float('nan')
    ).skb.set_name('finite_features')
    pred = features.skb.apply(
        NeuralFareRegressor(), y=y
    ).skb.set_name('model')
    return {
        'pred': pred,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys'],
    }