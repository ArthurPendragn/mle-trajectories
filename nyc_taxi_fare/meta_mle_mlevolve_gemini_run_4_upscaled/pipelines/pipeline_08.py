import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor

class ResBlock(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class WideAndDeepTabularResNet(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int=256, num_blocks: int=3):
        super().__init__()
        self.wide = nn.Linear(in_features, 1, bias=False)
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide(x)
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        h = self.gelu(h)
        deep_out = self.head(h)
        return wide_out + deep_out

class TabularResNetRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, hidden_dim: int=256, num_blocks: int=3, lr: float=0.003, epochs: int=25, batch_size: int=2048, random_state: int=42):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
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
        net = WideAndDeepTabularResNet(in_features=in_dim, hidden_dim=self.hidden_dim, num_blocks=self.num_blocks).to(device)
        with torch.no_grad():
            net.head.bias.fill_(y_mean)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=0.0001)
        dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_arr).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        total_steps = self.epochs * len(loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-05)
        criterion = nn.MSELoss()
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
                scheduler.step()
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
                batch_x = torch.from_numpy(X_scaled[i:i + 4096]).to(self.device_)
                batch_pred = self.net_(batch_x).squeeze(1).cpu().numpy()
                preds.append(batch_pred)
        return np.concatenate(preds)

class ResNetLGBMEnsembleRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, hidden_dim: int=256, num_blocks: int=3, lr: float=0.003, epochs: int=25, batch_size: int=2048, lgb_n_estimators: int=500, lgb_learning_rate: float=0.08, lgb_num_leaves: int=63, lgb_colsample_bytree: float=0.8, lgb_subsample: float=0.8, lgb_subsample_freq: int=1, weight_resnet: float=0.5, random_state: int=42, n_jobs: int=16):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lgb_n_estimators = lgb_n_estimators
        self.lgb_learning_rate = lgb_learning_rate
        self.lgb_num_leaves = lgb_num_leaves
        self.lgb_colsample_bytree = lgb_colsample_bytree
        self.lgb_subsample = lgb_subsample
        self.lgb_subsample_freq = lgb_subsample_freq
        self.weight_resnet = weight_resnet
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.resnet_ = TabularResNetRegressor(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size, random_state=self.random_state)
        self.resnet_.fit(X, y)
        self.lgb_ = LGBMRegressor(n_estimators=self.lgb_n_estimators, learning_rate=self.lgb_learning_rate, num_leaves=self.lgb_num_leaves, colsample_bytree=self.lgb_colsample_bytree, subsample=self.lgb_subsample, subsample_freq=self.lgb_subsample_freq, random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)
        self.lgb_.fit(X, y)
        return self

    def predict(self, X):
        pred_resnet = self.resnet_.predict(X)
        pred_lgb = self.lgb_.predict(X)
        w = self.weight_resnet
        return w * pred_resnet + (1.0 - w) * pred_lgb

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']
    coords = X[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]
    raw_d_lat = X['dropoff_latitude'] - X['pickup_latitude']
    raw_d_lon = X['dropoff_longitude'] - X['pickup_longitude']
    d_lat = raw_d_lat.abs()
    d_lon = raw_d_lon.abs()
    d_lat_km = d_lat * 111.13
    d_lon_km = d_lon * 84.18
    distance_km = (d_lat_km ** 2 + d_lon_km ** 2) ** 0.5
    manhattan_km = d_lat_km + d_lon_km
    theta = 29.0 * np.pi / 180.0
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    raw_dy_km = raw_d_lat * 111.13
    raw_dx_km = raw_d_lon * 84.18
    rot_dx = (raw_dx_km * cos_t - raw_dy_km * sin_t).abs()
    rot_dy = (raw_dx_km * sin_t + raw_dy_km * cos_t).abs()
    rotated_manhattan_km = rot_dx + rot_dy
    sin_bearing = raw_dx_km / (distance_km + 1e-05)
    cos_bearing = raw_dy_km / (distance_km + 1e-05)
    jfk_drop_dist = (((X['dropoff_latitude'] - 40.6413) * 111.13) ** 2 + ((X['dropoff_longitude'] - -73.7781) * 84.18) ** 2) ** 0.5
    lga_drop_dist = (((X['dropoff_latitude'] - 40.7769) * 111.13) ** 2 + ((X['dropoff_longitude'] - -73.874) * 84.18) ** 2) ** 0.5
    ewr_drop_dist = (((X['dropoff_latitude'] - 40.6895) * 111.13) ** 2 + ((X['dropoff_longitude'] - -74.1745) * 84.18) ** 2) ** 0.5
    jfk_pick_dist = (((X['pickup_latitude'] - 40.6413) * 111.13) ** 2 + ((X['pickup_longitude'] - -73.7781) * 84.18) ** 2) ** 0.5
    lga_pick_dist = (((X['pickup_latitude'] - 40.7769) * 111.13) ** 2 + ((X['pickup_longitude'] - -73.874) * 84.18) ** 2) ** 0.5
    ewr_pick_dist = (((X['pickup_latitude'] - 40.6895) * 111.13) ** 2 + ((X['pickup_longitude'] - -74.1745) * 84.18) ** 2) ** 0.5
    is_jfk = ((jfk_drop_dist < 2.5) | (jfk_pick_dist < 2.5)).astype(float)
    is_lga = ((lga_drop_dist < 2.5) | (lga_pick_dist < 2.5)).astype(float)
    is_ewr = ((ewr_drop_dist < 2.5) | (ewr_pick_dist < 2.5)).astype(float)
    midtown_pick_dist = (((X['pickup_latitude'] - 40.755) * 111.13) ** 2 + ((X['pickup_longitude'] - -73.985) * 84.18) ** 2) ** 0.5
    midtown_drop_dist = (((X['dropoff_latitude'] - 40.755) * 111.13) ** 2 + ((X['dropoff_longitude'] - -73.985) * 84.18) ** 2) ** 0.5
    is_pick_manhattan = ((X['pickup_longitude'] >= -74.02) & (X['pickup_longitude'] <= -73.93) & (X['pickup_latitude'] >= 40.7) & (X['pickup_latitude'] <= 40.82)).fillna(False)
    is_drop_manhattan = ((X['dropoff_longitude'] >= -74.02) & (X['dropoff_longitude'] <= -73.93) & (X['dropoff_latitude'] >= 40.7) & (X['dropoff_latitude'] <= 40.82)).fillna(False)
    is_cross_borough = (is_pick_manhattan != is_drop_manhattan).astype(float)
    is_jfk_manhattan = ((jfk_pick_dist < 2.5) & is_drop_manhattan | (jfk_drop_dist < 2.5) & is_pick_manhattan).astype(float)
    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek
    hour = dt.dt.hour
    hour_rad = hour * (2.0 * np.pi / 24.0)
    sin_hour = hour_rad.skb.apply_func(np.sin)
    cos_hour = hour_rad.skb.apply_func(np.cos)
    dow_rad = dayofweek * (2.0 * np.pi / 7.0)
    sin_dow = dow_rad.skb.apply_func(np.sin)
    cos_dow = dow_rad.skb.apply_func(np.cos)
    is_post_2012 = ((year > 2012) | (year == 2012) & (month >= 9)).fillna(False).astype(float)
    dist_x_post_2012 = distance_km * is_post_2012
    jfk_x_post_2012 = is_jfk * is_post_2012
    jfk_manhattan_x_post_2012 = is_jfk_manhattan * is_post_2012
    is_rush_hour = ((dayofweek < 5) & (hour >= 16) & (hour < 20)).fillna(False).astype(float)
    is_overnight = ((hour >= 20) | (hour < 6)).fillna(False).astype(float)
    features = coords.assign(d_lat=d_lat, d_lon=d_lon, distance_km=distance_km, manhattan_km=manhattan_km, rotated_manhattan_km=rotated_manhattan_km, sin_bearing=sin_bearing, cos_bearing=cos_bearing, jfk_drop_dist=jfk_drop_dist, lga_drop_dist=lga_drop_dist, ewr_drop_dist=ewr_drop_dist, jfk_pick_dist=jfk_pick_dist, lga_pick_dist=lga_pick_dist, ewr_pick_dist=ewr_pick_dist, is_jfk=is_jfk, is_lga=is_lga, is_ewr=is_ewr, midtown_pick_dist=midtown_pick_dist, midtown_drop_dist=midtown_drop_dist, is_cross_borough=is_cross_borough, is_jfk_manhattan=is_jfk_manhattan, jfk_manhattan_x_post_2012=jfk_manhattan_x_post_2012, year=year, month=month, day=day, dayofweek=dayofweek, hour=hour, sin_hour=sin_hour, cos_hour=cos_hour, sin_dow=sin_dow, cos_dow=cos_dow, is_post_2012=is_post_2012, dist_x_post_2012=dist_x_post_2012, jfk_x_post_2012=jfk_x_post_2012, is_rush_hour=is_rush_hour, is_overnight=is_overnight).skb.set_name('features')
    model = ResNetLGBMEnsembleRegressor(hidden_dim=256, num_blocks=3, lr=0.003, epochs=25, batch_size=2048, lgb_n_estimators=500, lgb_learning_rate=0.08, lgb_num_leaves=63, lgb_colsample_bytree=0.8, lgb_subsample=0.8, lgb_subsample_freq=1, weight_resnet=0.5, random_state=42, n_jobs=16)
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': scoring}