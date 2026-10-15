import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class ResNetBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class WideResNetModel(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_shortcut = nn.Linear(in_features, 1)
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )
        self.res1 = ResNetBlock(256)
        self.res2 = ResNetBlock(256)
        self.res3 = ResNetBlock(256)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        lin = self.linear_shortcut(x)
        h = self.in_proj(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        deep = self.head(h)
        return lin + deep

class TorchTaxiRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, lr=0.001, batch_size=2048, epochs=16, weight_decay=0.0001, seed=42):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        X_arr = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)
        y_arr = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y)
        y_arr = y_arr.astype(np.float32).reshape(-1, 1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ = device
        in_features = X_scaled.shape[1]
        self.net_ = WideResNetModel(in_features).to(device)
        dataset = TensorDataset(torch.from_numpy(X_scaled.astype(np.float32)), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-05)
        self.net_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = self.net_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, X):
        X_arr = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        X_scaled = self.scaler_.transform(X_arr)
        dataset = TensorDataset(torch.from_numpy(X_scaled.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)
        self.net_.eval()
        preds = []
        with torch.no_grad():
            for batch_x, in loader:
                batch_x = batch_x.to(self.device_)
                out = self.net_(batch_x)
                preds.append(out.cpu().numpy().ravel())
        preds = np.concatenate(preds)
        preds = np.clip(preds, 2.5, 500.0)
        return preds

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).predict(X)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    p_lon = X['pickup_longitude']
    p_lat = X['pickup_latitude']
    d_lon = X['dropoff_longitude']
    d_lat = X['dropoff_latitude']
    dx_km = (d_lon - p_lon) * 84.0
    dy_km = (d_lat - p_lat) * 111.0
    abs_dx = dx_km.abs()
    abs_dy = dy_km.abs()
    euclidean = (dx_km ** 2 + dy_km ** 2).pow(0.5)
    cardinal_manhattan = abs_dx + abs_dy
    safe_dist = euclidean + 1e-05
    bearing_x = dx_km / safe_dist
    bearing_y = dy_km / safe_dist
    theta = 29.0 * np.pi / 180.0
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rot_p_x = p_lon * 84.0 * cos_t - p_lat * 111.0 * sin_t
    rot_p_y = p_lon * 84.0 * sin_t + p_lat * 111.0 * cos_t
    rot_d_x = d_lon * 84.0 * cos_t - d_lat * 111.0 * sin_t
    rot_d_y = d_lon * 84.0 * sin_t + d_lat * 111.0 * cos_t
    rot_dx = (rot_d_x - rot_p_x).abs()
    rot_dy = (rot_d_y - rot_p_y).abs()
    rot_manhattan = rot_dx + rot_dy
    tortuosity = rot_manhattan / (euclidean + 0.05)
    is_zero_dist = (euclidean < 0.05).astype('float64')
    p_in_manhattan = ((p_lon >= -74.03) & (p_lon <= -73.93) & (p_lat >= 40.7) & (p_lat <= 40.88)).astype('float64')
    d_in_manhattan = ((d_lon >= -74.03) & (d_lon <= -73.93) & (d_lat >= 40.7) & (d_lat <= 40.88)).astype('float64')
    jfk_dist_p = (((p_lon - -73.7781) * 84.0) ** 2 + ((p_lat - 40.6413) * 111.0) ** 2).pow(0.5)
    jfk_dist_d = (((d_lon - -73.7781) * 84.0) ** 2 + ((d_lat - 40.6413) * 111.0) ** 2).pow(0.5)
    lga_dist_p = (((p_lon - -73.874) * 84.0) ** 2 + ((p_lat - 40.7769) * 111.0) ** 2).pow(0.5)
    lga_dist_d = (((d_lon - -73.874) * 84.0) ** 2 + ((d_lat - 40.7769) * 111.0) ** 2).pow(0.5)
    ewr_dist_p = (((p_lon - -74.1745) * 84.0) ** 2 + ((p_lat - 40.6895) * 111.0) ** 2).pow(0.5)
    ewr_d_dist = (((d_lon - -74.1745) * 84.0) ** 2 + ((d_lat - 40.6895) * 111.0) ** 2).pow(0.5)
    jfk_p_near = (jfk_dist_p <= 2.5).astype('float64')
    jfk_d_near = (jfk_dist_d <= 2.5).astype('float64')
    lga_p_near = (lga_dist_p <= 2.0).astype('float64')
    lga_d_near = (lga_dist_d <= 2.0).astype('float64')
    ewr_p_near = (ewr_dist_p <= 2.5).astype('float64')
    ewr_d_near = (ewr_d_dist <= 2.5).astype('float64')
    p_man_d_jfk = p_in_manhattan * jfk_d_near
    p_jfk_d_man = jfk_p_near * d_in_manhattan
    p_man_d_ewr = p_in_manhattan * ewr_d_near
    p_ewr_d_man = ewr_p_near * d_in_manhattan
    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    year_raw = dt.dt.year.astype('float64')
    month_raw = dt.dt.month.astype('float64')
    day_raw = dt.dt.day.astype('float64')
    hour = dt.dt.hour.astype('float64')
    dayofweek = dt.dt.dayofweek.astype('float64')
    post_hike = ((year_raw > 2012.0) | (year_raw == 2012.0) & ((month_raw > 9.0) | (month_raw == 9.0) & (day_raw >= 4.0))).astype('float64')
    rot_man_post_hike = rot_manhattan * post_hike
    jfk_flat_man = (p_man_d_jfk + p_jfk_d_man).clip(0.0, 1.0)
    jfk_flat_post_hike = jfk_flat_man * post_hike
    jfk_flat_pre_hike = jfk_flat_man * (1.0 - post_hike)
    is_rush_hour = ((dayofweek < 5.0) & (hour >= 16.0) & (hour < 20.0)).astype('float64')
    is_overnight = ((hour >= 20.0) | (hour < 6.0)).astype('float64')
    frac_year = (year_raw + (month_raw - 1.0) / 12.0 + (day_raw - 1.0) / 365.0 - 2009.0) / 6.0
    hour_rad = hour * (2.0 * np.pi / 24.0)
    hour_sin = hour_rad.skb.apply_func(np.sin)
    hour_cos = hour_rad.skb.apply_func(np.cos)
    dow_rad = dayofweek * (2.0 * np.pi / 7.0)
    dow_sin = dow_rad.skb.apply_func(np.sin)
    dow_cos = dow_rad.skb.apply_func(np.cos)
    month_rad = (month_raw - 1.0) * (2.0 * np.pi / 12.0)
    month_sin = month_rad.skb.apply_func(np.sin)
    month_cos = month_rad.skb.apply_func(np.cos)
    norm_p_lon = (p_lon - -73.97) * 10.0
    norm_p_lat = (p_lat - 40.75) * 10.0
    norm_d_lon = (d_lon - -73.97) * 10.0
    norm_d_lat = (d_lat - 40.75) * 10.0
    fourier_dict = {}
    coords = {'p_lon': norm_p_lon, 'p_lat': norm_p_lat, 'd_lon': norm_d_lon, 'd_lat': norm_d_lat}
    for c_name, c_val in coords.items():
        for freq in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
            f_arg = c_val * (freq * np.pi)
            f_tag = int(freq)
            fourier_dict[f'{c_name}_sin_{f_tag}'] = f_arg.skb.apply_func(np.sin)
            fourier_dict[f'{c_name}_cos_{f_tag}'] = f_arg.skb.apply_func(np.cos)
    feature_dict = {
        'pickup_longitude': p_lon,
        'pickup_latitude': p_lat,
        'dropoff_longitude': d_lon,
        'dropoff_latitude': d_lat,
        'dx_km': dx_km,
        'dy_km': dy_km,
        'abs_dx': abs_dx,
        'abs_dy': abs_dy,
        'euclidean': euclidean,
        'cardinal_manhattan': cardinal_manhattan,
        'bearing_x': bearing_x,
        'bearing_y': bearing_y,
        'rot_dx': rot_dx,
        'rot_dy': rot_dy,
        'rot_manhattan': rot_manhattan,
        'tortuosity': tortuosity,
        'is_zero_dist': is_zero_dist,
        'p_in_manhattan': p_in_manhattan,
        'd_in_manhattan': d_in_manhattan,
        'p_man_d_jfk': p_man_d_jfk,
        'p_jfk_d_man': p_jfk_d_man,
        'jfk_flat_post_hike': jfk_flat_post_hike,
        'jfk_flat_pre_hike': jfk_flat_pre_hike,
        'p_man_d_ewr': p_man_d_ewr,
        'p_ewr_d_man': p_ewr_d_man,
        'jfk_dist_p': jfk_dist_p,
        'jfk_dist_d': jfk_dist_d,
        'jfk_p_near': jfk_p_near,
        'jfk_d_near': jfk_d_near,
        'lga_dist_p': lga_dist_p,
        'lga_dist_d': lga_dist_d,
        'lga_p_near': lga_p_near,
        'lga_d_near': lga_d_near,
        'ewr_dist_p': ewr_dist_p,
        'ewr_d_dist': ewr_d_dist,
        'ewr_p_near': ewr_p_near,
        'ewr_d_near': ewr_d_near,
        'frac_year': frac_year,
        'post_hike': post_hike,
        'rot_man_post_hike': rot_man_post_hike,
        'is_rush_hour': is_rush_hour,
        'is_overnight': is_overnight,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'month_sin': month_sin,
        'month_cos': month_cos
    }
    feature_dict.update(fourier_dict)
    features = X[['passenger_count']].astype('float64').assign(**feature_dict).skb.set_name('features')
    model = TorchTaxiRegressor(lr=0.001, batch_size=2048, epochs=16)
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}