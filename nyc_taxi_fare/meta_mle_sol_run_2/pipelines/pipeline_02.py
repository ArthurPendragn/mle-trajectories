import numpy as np
import pandas as pd
import skrub
from torch.nn import Module
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


class FareNetwork(Module):
    def __init__(self, width):
        from torch import nn

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class TensorRows(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, indices):
        return self.X[indices], self.y[indices]


class NeuralFareRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, epochs=10, batch_size=8192, learning_rate=0.001,
                 weight_decay=0.0001, random_state=1971):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _prepare(self, X):
        values = np.asarray(X, dtype=np.float32).copy()
        for index in self.log_indices_:
            column = values[:, index]
            values[:, index] = np.sign(column) * np.log1p(np.abs(column))
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, RandomSampler, BatchSampler

        torch.manual_seed(self.random_state)
        torch.set_num_threads(8)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cuda.matmul.allow_tf32 = True
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_features_in_ = X.shape[1]
        self.log_indices_ = [
            i for i, name in enumerate(X.columns)
            if name in ('dx', 'dy', 'distance', 'manhattan')
        ]
        values = self._prepare(X)
        self.scaler_ = StandardScaler()
        values = self.scaler_.fit_transform(values)
        np.clip(values, -10, 10, out=values)
        targets = np.asarray(y, dtype=np.float32)
        inputs = torch.as_tensor(values, device=self.device_)
        labels = torch.as_tensor(targets, device=self.device_)
        dataset = TensorRows(inputs, labels)
        generator = torch.Generator().manual_seed(self.random_state)
        batches = BatchSampler(
            RandomSampler(dataset, generator=generator),
            batch_size=self.batch_size, drop_last=False
        )
        loader = DataLoader(dataset, sampler=batches, batch_size=None,
                            num_workers=0)
        self.net_ = FareNetwork(self.n_features_in_).to(self.device_)
        with torch.no_grad():
            self.net_.layers[-1].bias.fill_(float(np.mean(targets)))
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        self.net_.train()
        for epoch in range(self.epochs):
            loss_sum = torch.zeros((), device=self.device_)
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = self.net_(xb)
                loss = nn.functional.mse_loss(prediction, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach() * len(xb)
            scheduler.step()
            print("epoch", epoch + 1, "training_mse",
                  float(loss_sum / len(dataset)), flush=True)
        return self

    def predict(self, X):
        import torch

        values = self.scaler_.transform(self._prepare(X))
        np.clip(values, -10, 10, out=values)
        self.net_.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = torch.as_tensor(
                    values[start:start + self.batch_size],
                    device=self.device_
                )
                outputs.append(self.net_(batch).cpu().numpy())
        return np.concatenate(outputs)


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    dt = X['pickup_datetime'].skb.apply_func(
        pd.to_datetime, utc=True, errors='coerce'
    )
    calendar = X[['passenger_count']].assign(
        year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
        weekday=dt.dt.dayofweek, hour=dt.dt.hour, minute=dt.dt.minute
    ).astype('float32').skb.set_name('calendar')
    dx = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.4
    dy = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.2
    geography = X[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]].assign(
        dx=dx, dy=dy, distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs()
    ).astype('float32').skb.set_name('geography')
    features = geography.skb.concat(
        [calendar], axis=1
    ).replace([np.inf, -np.inf], np.nan).skb.set_name('finite_features')
    pred = features.skb.apply(
        NeuralFareRegressor(), y=y
    ).skb.set_name('model')
    result = {'pred': pred, 'scoring': setup['scoring']}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result