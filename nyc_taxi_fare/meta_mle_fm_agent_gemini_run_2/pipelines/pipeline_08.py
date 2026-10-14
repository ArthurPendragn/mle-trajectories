import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class ResBlock(nn.Module):

    def __init__(self, dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=192, n_blocks=2, dropout=0.05):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

class TorchTabularResNet(BaseEstimator, RegressorMixin):

    def __init__(self, hidden_dim=192, n_blocks=2, dropout=0.05, lr=0.002, weight_decay=1e-05, batch_size=4096, epochs=15, seed=42):
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_ = StandardScaler()
        X_mat = self.scaler_.fit_transform(X)
        X_mat = np.nan_to_num(X_mat, nan=0.0).astype(np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        X_t = torch.from_numpy(X_mat)
        y_t = torch.from_numpy(y_arr)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        model = TabularResNet(in_features=X_mat.shape[1], hidden_dim=self.hidden_dim, n_blocks=self.n_blocks, dropout=self.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
            scheduler.step()
        self.model_ = model
        self.device_ = device
        return self

    def predict(self, X):
        X_mat = self.scaler_.transform(X)
        X_mat = np.nan_to_num(X_mat, nan=0.0).astype(np.float32)
        X_t = torch.from_numpy(X_mat)
        loader = DataLoader(X_t, batch_size=self.batch_size * 2, shuffle=False)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for bx in loader:
                bx = bx.to(self.device_)
                pred = self.model_(bx)
                preds.append(pred.cpu().numpy())
        res = np.concatenate(preds, axis=0)
        return np.clip(res, 2.5, 500.0)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, format='%Y-%m-%d %H:%M:%S UTC')
    year = dt.dt.year.astype('float32')
    month = dt.dt.month.astype('float32')
    day = dt.dt.day.astype('float32')
    hour = dt.dt.hour.astype('float32')
    weekday = dt.dt.dayofweek.astype('float32')
    d_lon = (X['dropoff_longitude'] - X['pickup_longitude']).abs()
    d_lat = (X['dropoff_latitude'] - X['pickup_latitude']).abs()
    dx_km = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.0
    dy_km = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.0
    euclidean_dist = (dy_km ** 2 + dx_km ** 2) ** 0.5
    p_lat_rad = X['pickup_latitude'] * (np.pi / 180.0)
    d_lat_rad = X['dropoff_latitude'] * (np.pi / 180.0)
    d_lon_rad = (X['dropoff_longitude'] - X['pickup_longitude']) * (np.pi / 180.0)
    d_lat_diff = (X['dropoff_latitude'] - X['pickup_latitude']) * (np.pi / 180.0)
    sin_dlat = (d_lat_diff * 0.5).skb.apply_func(np.sin)
    sin_dlon = (d_lon_rad * 0.5).skb.apply_func(np.sin)
    cos_plat = p_lat_rad.skb.apply_func(np.cos)
    cos_dlat = d_lat_rad.skb.apply_func(np.cos)
    a = sin_dlat * sin_dlat + cos_plat * cos_dlat * sin_dlon * sin_dlon
    a_clipped = a.skb.apply_func(np.clip, 0.0, 1.0)
    haversine_dist = a_clipped.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin) * (2.0 * 6371.0)
    theta = 29.0 * (np.pi / 180.0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    dx_rot = dx_km * cos_theta - dy_km * sin_theta
    dy_rot = dx_km * sin_theta + dy_km * cos_theta
    rotated_manhattan_dist = dx_rot.abs() + dy_rot.abs()
    bearing = dy_km.skb.apply_func(np.arctan2, dx_km)
    dist_pickup_jfk = (((X['pickup_latitude'] - 40.6413) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.7781) * 84.0) ** 2) ** 0.5
    dist_dropoff_jfk = (((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.7781) * 84.0) ** 2) ** 0.5
    dist_pickup_lga = (((X['pickup_latitude'] - 40.7769) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.874) * 84.0) ** 2) ** 0.5
    dist_dropoff_lga = (((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.874) * 84.0) ** 2) ** 0.5
    dist_pickup_ewr = (((X['pickup_latitude'] - 40.6895) * 111.0) ** 2 + ((X['pickup_longitude'] - -74.1745) * 84.0) ** 2) ** 0.5
    dist_dropoff_ewr = (((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2 + ((X['dropoff_longitude'] - -74.1745) * 84.0) ** 2) ** 0.5
    dist_pickup_midtown = (((X['pickup_latitude'] - 40.758) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.9855) * 84.0) ** 2) ** 0.5
    dist_dropoff_midtown = (((X['dropoff_latitude'] - 40.758) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.9855) * 84.0) ** 2) ** 0.5
    fare_hike_2012 = ((year > 2012) | (year == 2012) & (month >= 9)).astype('float32')
    rush_hour_surcharge = ((weekday < 5) & (hour >= 16) & (hour < 20)).astype('float32')
    overnight_surcharge = ((hour >= 20) | (hour < 6)).astype('float32')
    features = X.drop(columns=['key', 'pickup_datetime']).assign(
        abs_diff_lon=d_lon,
        abs_diff_lat=d_lat,
        euclidean_dist=euclidean_dist,
        haversine_dist=haversine_dist,
        rotated_manhattan_dist=rotated_manhattan_dist,
        bearing=bearing,
        dist_pickup_jfk=dist_pickup_jfk,
        dist_dropoff_jfk=dist_dropoff_jfk,
        dist_pickup_lga=dist_pickup_lga,
        dist_dropoff_lga=dist_dropoff_lga,
        dist_pickup_ewr=dist_pickup_ewr,
        dist_dropoff_ewr=dist_dropoff_ewr,
        dist_pickup_midtown=dist_pickup_midtown,
        dist_dropoff_midtown=dist_dropoff_midtown,
        year=year,
        month=month,
        day=day,
        hour=hour,
        weekday=weekday,
        fare_hike_2012=fare_hike_2012,
        rush_hour_surcharge=rush_hour_surcharge,
        overnight_surcharge=overnight_surcharge,
    ).skb.set_name('features')
    model = skrub.choose_from({
        'dim192': TorchTabularResNet(hidden_dim=192, n_blocks=2),
        'dim256': TorchTabularResNet(hidden_dim=256, n_blocks=2),
    }, name='resnet_capacity')
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}