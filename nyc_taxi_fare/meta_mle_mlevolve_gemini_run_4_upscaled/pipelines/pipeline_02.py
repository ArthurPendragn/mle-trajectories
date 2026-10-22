import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class TorchMLPRegressor(RegressorMixin, BaseEstimator):

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 3e-3,
        epochs: int = 15,
        batch_size: int = 2048,
        random_state: int = 42,
    ):
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
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        X_mat = np.asarray(X, dtype=np.float32)
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_mat).astype(np.float32)

        y_arr = np.asarray(y, dtype=np.float32).ravel()
        y_mean = float(y_arr.mean())

        in_dim = X_scaled.shape[1]

        class MLP(nn.Module):

            def __init__(self, in_features, hidden):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, hidden // 2),
                    nn.BatchNorm1d(hidden // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden // 2, hidden // 4),
                    nn.BatchNorm1d(hidden // 4),
                    nn.GELU(),
                    nn.Linear(hidden // 4, 1),
                )

            def forward(self, x):
                return self.net(x)

        net = MLP(in_dim, self.hidden_dim).to(device)
        with torch.no_grad():
            last_linear = net.net[-1]
            last_linear.bias.fill_(y_mean)

        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.lr, weight_decay=1e-4
        )
        criterion = nn.MSELoss()

        dataset = TensorDataset(
            torch.from_numpy(X_scaled), torch.from_numpy(y_arr).unsqueeze(1)
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        net.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                out = net(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        self.net_ = net
        self.device_ = device
        return self

    def predict(self, X):
        X_mat = np.asarray(X, dtype=np.float32)
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler_.transform(X_mat).astype(np.float32)

        if len(X_scaled) == 0:
            return np.array([], dtype=np.float32)

        self.net_.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), 4096):
                batch_x = torch.from_numpy(X_scaled[i : i + 4096]).to(
                    self.device_
                )
                batch_pred = self.net_(batch_x).squeeze(1).cpu().numpy()
                preds.append(batch_pred)
        return np.concatenate(preds)


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    coords = X[[
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count',
    ]]

    d_lat = (X['dropoff_latitude'] - X['pickup_latitude']).abs()
    d_lon = (X['dropoff_longitude'] - X['pickup_longitude']).abs()
    d_lat_km = d_lat * 111.13
    d_lon_km = d_lon * 84.18
    distance_km = (d_lat_km**2 + d_lon_km**2) ** 0.5
    manhattan_km = d_lat_km + d_lon_km

    jfk_drop_dist = (
        ((X['dropoff_latitude'] - 40.6413) * 111.13) ** 2
        + ((X['dropoff_longitude'] - -73.7781) * 84.18) ** 2
    ) ** 0.5
    lga_drop_dist = (
        ((X['dropoff_latitude'] - 40.7769) * 111.13) ** 2
        + ((X['dropoff_longitude'] - -73.874) * 84.18) ** 2
    ) ** 0.5
    ewr_drop_dist = (
        ((X['dropoff_latitude'] - 40.6895) * 111.13) ** 2
        + ((X['dropoff_longitude'] - -74.1745) * 84.18) ** 2
    ) ** 0.5

    jfk_pick_dist = (
        ((X['pickup_latitude'] - 40.6413) * 111.13) ** 2
        + ((X['pickup_longitude'] - -73.7781) * 84.18) ** 2
    ) ** 0.5
    lga_pick_dist = (
        ((X['pickup_latitude'] - 40.7769) * 111.13) ** 2
        + ((X['pickup_longitude'] - -73.874) * 84.18) ** 2
    ) ** 0.5
    ewr_pick_dist = (
        ((X['pickup_latitude'] - 40.6895) * 111.13) ** 2
        + ((X['pickup_longitude'] - -74.1745) * 84.18) ** 2
    ) ** 0.5

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek
    hour = dt.dt.hour

    features = coords.assign(
        d_lat=d_lat,
        d_lon=d_lon,
        distance_km=distance_km,
        manhattan_km=manhattan_km,
        jfk_drop_dist=jfk_drop_dist,
        lga_drop_dist=lga_drop_dist,
        ewr_drop_dist=ewr_drop_dist,
        jfk_pick_dist=jfk_pick_dist,
        lga_pick_dist=lga_pick_dist,
        ewr_pick_dist=ewr_pick_dist,
        year=year,
        month=month,
        day=day,
        dayofweek=dayofweek,
        hour=hour,
    ).skb.set_name('features')

    model = TorchMLPRegressor(
        hidden_dim=256, lr=3e-3, epochs=15, batch_size=2048, random_state=42
    )
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': scoring}