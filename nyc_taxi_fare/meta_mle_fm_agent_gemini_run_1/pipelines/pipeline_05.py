import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class PyTorchMLP(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchMLPRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, epochs=15, batch_size=2048, lr=0.003, weight_decay=0.0001, seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.scaler_ = StandardScaler()
        X_arr = self.scaler_.fit_transform(np.asarray(X, dtype=np.float32))
        y_arr = np.asarray(y, dtype=np.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ = device
        in_features = X_arr.shape[1]
        self.model_ = PyTorchMLP(in_features).to(device)
        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs * len(loader))
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
                scheduler.step()
        return self

    def predict(self, X):
        X_arr = self.scaler_.transform(np.asarray(X, dtype=np.float32))
        self.model_.eval()
        device = self.device_
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_arr), self.batch_size * 2):
                batch_x = torch.from_numpy(X_arr[i:i + self.batch_size * 2]).to(device)
                batch_pred = self.model_(batch_x)
                preds.append(batch_pred.cpu().numpy())
        return np.concatenate(preds)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    year = X['pickup_datetime'].str.slice(0, 4).astype('float32')
    month = X['pickup_datetime'].str.slice(5, 7).astype('float32')
    day = X['pickup_datetime'].str.slice(8, 10).astype('float32')
    hour = X['pickup_datetime'].str.slice(11, 13).astype('float32')

    month_offset = month.astype('int32').map({
        1: 0, 2: 3, 3: 2, 4: 5, 5: 0, 6: 3, 7: 5, 8: 1, 9: 4, 10: 6, 11: 2, 12: 4
    }).astype('float32')
    y_adj = year.astype('int32') - (month < 3).fillna(False).astype('int32')
    day_of_week = ((y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + month_offset.astype('int32') + day.astype('int32')) % 7).astype('float32')
    is_weekend = ((day_of_week == 0) | (day_of_week == 6)).fillna(False).astype('float32')

    hour_sin = (hour * float(2 * np.pi / 24.0)).skb.apply_func(np.sin)
    hour_cos = (hour * float(2 * np.pi / 24.0)).skb.apply_func(np.cos)
    dow_sin = (day_of_week * float(2 * np.pi / 7.0)).skb.apply_func(np.sin)
    dow_cos = (day_of_week * float(2 * np.pi / 7.0)).skb.apply_func(np.cos)
    month_sin = (month * float(2 * np.pi / 12.0)).skb.apply_func(np.sin)
    month_cos = (month * float(2 * np.pi / 12.0)).skb.apply_func(np.cos)

    dlon = X['dropoff_longitude'] - X['pickup_longitude']
    dlat = X['dropoff_latitude'] - X['pickup_latitude']
    abs_dlon = dlon.abs()
    abs_dlat = dlat.abs()
    mid_lon = (X['pickup_longitude'] + X['dropoff_longitude']) * 0.5
    mid_lat = (X['pickup_latitude'] + X['dropoff_latitude']) * 0.5

    euclidean_dist = ((abs_dlon * 0.75756) ** 2 + abs_dlat ** 2) ** 0.5 * 111.0
    manhattan_dist = abs_dlon * 84.09 + abs_dlat * 111.0

    p_jfk = (((X['pickup_longitude'] - -73.7781) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.6413) ** 2) ** 0.5 * 111.0
    d_jfk = (((X['dropoff_longitude'] - -73.7781) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.6413) ** 2) ** 0.5 * 111.0
    p_lga = (((X['pickup_longitude'] - -73.874) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.7769) ** 2) ** 0.5 * 111.0
    d_lga = (((X['dropoff_longitude'] - -73.874) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.7769) ** 2) ** 0.5 * 111.0
    p_ewr = (((X['pickup_longitude'] - -74.1745) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.6895) ** 2) ** 0.5 * 111.0
    d_ewr = (((X['dropoff_longitude'] - -74.1745) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.6895) ** 2) ** 0.5 * 111.0
    p_mid = (((X['pickup_longitude'] - -73.9855) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.758) ** 2) ** 0.5 * 111.0
    d_mid = (((X['dropoff_longitude'] - -73.9855) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.758) ** 2) ** 0.5 * 111.0

    features = X[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']].assign(
        year=year,
        month=month,
        day=day,
        hour=hour,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        dow_sin=dow_sin,
        dow_cos=dow_cos,
        month_sin=month_sin,
        month_cos=month_cos,
        dlon=dlon,
        dlat=dlat,
        mid_lon=mid_lon,
        mid_lat=mid_lat,
        abs_dlon=abs_dlon,
        abs_dlat=abs_dlat,
        euclidean_km=euclidean_dist,
        manhattan_km=manhattan_dist,
        p_jfk_km=p_jfk,
        d_jfk_km=d_jfk,
        p_lga_km=p_lga,
        d_lga_km=d_lga,
        p_ewr_km=p_ewr,
        d_ewr_km=d_ewr,
        p_mid_km=p_mid,
        d_mid_km=d_mid,
    ).skb.set_name('features')

    model = TorchMLPRegressor(epochs=15, batch_size=2048, lr=0.003, weight_decay=0.0001, seed=42)
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}