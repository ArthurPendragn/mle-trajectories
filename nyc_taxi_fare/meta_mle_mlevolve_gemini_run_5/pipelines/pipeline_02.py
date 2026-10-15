import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import skrub


class PyTorchMLP(nn.Module):

    def __init__(self, in_features, hidden_dim=256, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PyTorchMLPRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, hidden_dim=256, lr=1e-3, epochs=15, batch_size=2048, random_state=42):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.scaler_ = StandardScaler()
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.to_numpy(dtype=np.float32).ravel()
        else:
            y_arr = np.asarray(y, dtype=np.float32).ravel()

        X_scaled = self.scaler_.fit_transform(X_arr)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_features = X_scaled.shape[1]
        self.model_ = PyTorchMLP(in_features=in_features, hidden_dim=self.hidden_dim).to(device)

        dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                pred = self.model_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        X_scaled = self.scaler_.transform(X_arr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), self.batch_size):
                batch_x = torch.from_numpy(X_scaled[i : i + self.batch_size]).to(device)
                p = self.model_(batch_x)
                preds.append(p.cpu().numpy())

        return np.concatenate(preds, axis=0)


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    p_lat = X['pickup_latitude']
    p_lon = X['pickup_longitude']
    d_lat = X['dropoff_latitude']
    d_lon = X['dropoff_longitude']

    d_lat_abs = (d_lat - p_lat).abs()
    d_lon_abs = (d_lon - p_lon).abs()
    d_lat_signed = d_lat - p_lat
    d_lon_signed = d_lon - p_lon

    lat_km = d_lat_abs * 111.0
    lon_km = d_lon_abs * 84.3
    euclidean_km = (lat_km**2 + lon_km**2) ** 0.5
    manhattan_km = lat_km + lon_km

    p_jfk = (((p_lat - 40.6413) * 111.0) ** 2 + ((p_lon - -73.7781) * 84.3) ** 2) ** 0.5
    d_jfk = (((d_lat - 40.6413) * 111.0) ** 2 + ((d_lon - -73.7781) * 84.3) ** 2) ** 0.5

    p_lga = (((p_lat - 40.7769) * 111.0) ** 2 + ((p_lon - -73.874) * 84.3) ** 2) ** 0.5
    d_lga = (((d_lat - 40.7769) * 111.0) ** 2 + ((d_lon - -73.874) * 84.3) ** 2) ** 0.5

    p_ewr = (((p_lat - 40.6895) * 111.0) ** 2 + ((p_lon - -74.1745) * 84.3) ** 2) ** 0.5
    d_ewr = (((d_lat - 40.6895) * 111.0) ** 2 + ((d_lon - -74.1745) * 84.3) ** 2) ** 0.5

    p_midtown = (((p_lat - 40.7580) * 111.0) ** 2 + ((p_lon - -73.9855) * 84.3) ** 2) ** 0.5
    d_midtown = (((d_lat - 40.7580) * 111.0) ** 2 + ((d_lon - -73.9855) * 84.3) ** 2) ** 0.5

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    hour = dt.dt.hour
    weekday = dt.dt.dayofweek

    features = X[
        ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    ].assign(
        d_lat_abs=d_lat_abs,
        d_lon_abs=d_lon_abs,
        d_lat=d_lat_signed,
        d_lon=d_lon_signed,
        lat_km=lat_km,
        lon_km=lon_km,
        euclidean_km=euclidean_km,
        manhattan_km=manhattan_km,
        p_jfk=p_jfk,
        d_jfk=d_jfk,
        p_lga=p_lga,
        d_lga=d_lga,
        p_ewr=p_ewr,
        d_ewr=d_ewr,
        p_midtown=p_midtown,
        d_midtown=d_midtown,
        year=year,
        month=month,
        day=day,
        hour=hour,
        weekday=weekday,
    ).skb.set_name('features')

    model = skrub.choose_from(
        {
            'mlp_hidden256': PyTorchMLPRegressor(
                hidden_dim=256, lr=1e-3, epochs=15, batch_size=2048, random_state=42
            ),
            'mlp_hidden512': PyTorchMLPRegressor(
                hidden_dim=512, lr=1e-3, epochs=15, batch_size=2048, random_state=42
            ),
        },
        name='model_variant',
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': scoring}