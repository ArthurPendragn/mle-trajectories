import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

class ResNetBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Dropout(0.05), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.block(x)

class WideResNetModel(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.linear_shortcut = nn.Linear(in_features, 1)
        self.in_proj = nn.Sequential(nn.Linear(in_features, 256), nn.LayerNorm(256), nn.SiLU())
        self.res1 = ResNetBlock(256)
        self.res2 = ResNetBlock(256)
        self.res3 = ResNetBlock(256)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 1))

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

class TriBlendedTaxiRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, lgb_weight=0.38, xgb_weight=0.38, torch_weight=0.24, lgb_trees=700, lgb_lr=0.06, lgb_leaves=45, lgb_min_child=50, lgb_reg_lambda=1.0, xgb_trees=850, xgb_lr=0.05, xgb_reg_lambda=2.0, xgb_min_child=20.0, torch_epochs=16, batch_size=2048, lr=0.001, seed=42):
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        self.torch_weight = torch_weight
        self.lgb_trees = lgb_trees
        self.lgb_lr = lgb_lr
        self.lgb_leaves = lgb_leaves
        self.lgb_min_child = lgb_min_child
        self.lgb_reg_lambda = lgb_reg_lambda
        self.xgb_trees = xgb_trees
        self.xgb_lr = xgb_lr
        self.xgb_reg_lambda = xgb_reg_lambda
        self.xgb_min_child = xgb_min_child
        self.torch_epochs = torch_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

    def fit(self, X, y):
        fourier_prefixes = ('p_lon_', 'p_lat_', 'd_lon_', 'd_lat_', 'm_lon_', 'm_lat_')
        if hasattr(X, 'columns'):
            self.tree_cols_ = [c for c in X.columns if not any((c.startswith(p) for p in fourier_prefixes))]
            X_tree = X[self.tree_cols_]
        else:
            self.tree_cols_ = None
            X_tree = X
        self.lgb_ = LGBMRegressor(objective='regression', n_estimators=self.lgb_trees, learning_rate=self.lgb_lr, num_leaves=self.lgb_leaves, min_child_samples=self.lgb_min_child, reg_lambda=self.lgb_reg_lambda, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=self.seed, n_jobs=8)
        self.lgb_.fit(X_tree, y)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xgb_ = XGBRegressor(objective='reg:squarederror', n_estimators=self.xgb_trees, learning_rate=self.xgb_lr, max_depth=8, reg_lambda=self.xgb_reg_lambda, min_child_weight=self.xgb_min_child, subsample=0.8, colsample_bytree=0.8, tree_method='hist', device=device, random_state=self.seed, n_jobs=8)
        self.xgb_.fit(X_tree, y)
        self.torch_ = TorchTaxiRegressor(lr=self.lr, batch_size=self.batch_size, epochs=self.torch_epochs, seed=self.seed)
        self.torch_.fit(X, y)
        return self

    def predict(self, X):
        X_tree = X[self.tree_cols_] if self.tree_cols_ is not None and hasattr(X, '__getitem__') else X
        pred_lgb = self.lgb_.predict(X_tree)
        pred_xgb = self.xgb_.predict(X_tree)
        pred_torch = self.torch_.predict(X)
        blend = self.lgb_weight * pred_lgb + self.xgb_weight * pred_xgb + self.torch_weight * pred_torch
        return np.clip(blend, 2.5, 500.0)

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
    mid_lon = (p_lon + d_lon) * 0.5
    mid_lat = (p_lat + d_lat) * 0.5
    dx_km = (d_lon - p_lon) * 84.0
    dy_km = (d_lat - p_lat) * 111.0
    abs_dx = dx_km.abs()
    abs_dy = dy_km.abs()
    ratio_dx_dy = abs_dx / (abs_dy + 0.01)
    euclidean = (dx_km ** 2 + dy_km ** 2).pow(0.5)
    log_euclidean = euclidean.skb.apply_func(np.log1p)
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
    log_rot_manhattan = rot_manhattan.skb.apply_func(np.log1p)
    excess_manhattan = (rot_manhattan - euclidean).clip(lower=0.0)
    tortuosity = rot_manhattan / (euclidean + 0.05)
    is_zero_dist = (euclidean < 0.05).astype('float64')
    mid_rot_x = (rot_p_x + rot_d_x) * 0.5
    mid_rot_y = (rot_p_y + rot_d_y) * 0.5
    p_in_manhattan = ((p_lon >= -74.03) & (p_lon <= -73.93) & (p_lat >= 40.7) & (p_lat <= 40.88)).astype('float64')
    d_in_manhattan = ((d_lon >= -74.03) & (d_lon <= -73.93) & (d_lat >= 40.7) & (d_lat <= 40.88)).astype('float64')
    intra_manhattan = p_in_manhattan * d_in_manhattan
    cross_manhattan = (p_in_manhattan + d_in_manhattan == 1.0).astype('float64')
    p_in_brooklyn = ((p_lon >= -74.05) & (p_lon <= -73.85) & (p_lat >= 40.55) & (p_lat <= 40.74)).astype('float64')
    d_in_brooklyn = ((d_lon >= -74.05) & (d_lon <= -73.85) & (d_lat >= 40.55) & (d_lat <= 40.74)).astype('float64')
    p_in_queens = ((p_lon >= -73.96) & (p_lon <= -73.7) & (p_lat >= 40.55) & (p_lat <= 40.8)).astype('float64')
    d_in_queens = ((d_lon >= -73.96) & (d_lon <= -73.7) & (d_lat >= 40.55) & (d_lat <= 40.8)).astype('float64')
    p_in_bronx = ((p_lon >= -73.93) & (p_lon <= -73.75) & (p_lat >= 40.8) & (p_lat <= 40.92)).astype('float64')
    d_in_bronx = ((d_lon >= -73.93) & (d_lon <= -73.75) & (d_lat >= 40.8) & (d_lat <= 40.92)).astype('float64')
    p_in_si = ((p_lon >= -74.26) & (p_lon <= -74.05) & (p_lat >= 40.5) & (p_lat <= 40.65)).astype('float64')
    d_in_si = ((d_lon >= -74.26) & (d_lon <= -74.05) & (d_lat >= 40.5) & (d_lat <= 40.65)).astype('float64')
    si_trip = (p_in_si + d_in_si).clip(0.0, 1.0)
    cross_si_dist = si_trip * euclidean
    dropoff_in_nassau = ((d_lon > -73.7) & (d_lat >= 40.55) & (d_lat <= 40.92)).astype('float64')
    nassau_dist = dropoff_in_nassau * euclidean
    dropoff_in_westchester = (d_lat > 40.915).astype('float64')
    westchester_dist = dropoff_in_westchester * euclidean
    dropoff_in_nj = ((d_lon < -74.04) & (d_lat >= 40.7) & (d_lat <= 41.1) & (p_in_si == 0.0) & (d_in_si == 0.0)).astype('float64')
    nj_dist = dropoff_in_nj * euclidean
    p_man_d_bk = p_in_manhattan * d_in_brooklyn
    p_bk_d_man = p_in_brooklyn * d_in_manhattan
    p_man_d_qn = p_in_manhattan * d_in_queens
    p_qn_d_man = p_in_queens * d_in_manhattan
    p_man_d_bx = p_in_manhattan * d_in_bronx
    p_bx_d_man = p_in_bronx * d_in_manhattan
    cross_bk_dist = (p_man_d_bk + p_bk_d_man) * euclidean
    cross_qn_dist = (p_man_d_qn + p_qn_d_man) * euclidean
    cross_bx_dist = (p_man_d_bx + p_bx_d_man) * euclidean
    cross_bk_qn = (p_in_brooklyn * d_in_queens + p_in_queens * d_in_brooklyn).clip(0.0, 1.0)
    cross_bk_qn_dist = cross_bk_qn * euclidean
    midtown_dist_p = (((p_lon - -73.985) * 84.0) ** 2 + ((p_lat - 40.755) * 111.0) ** 2).pow(0.5)
    midtown_dist_d = (((d_lon - -73.985) * 84.0) ** 2 + ((d_lat - 40.755) * 111.0) ** 2).pow(0.5)
    downtown_dist_p = (((p_lon - -74.006) * 84.0) ** 2 + ((p_lat - 40.7128) * 111.0) ** 2).pow(0.5)
    downtown_dist_d = (((d_lon - -74.006) * 84.0) ** 2 + ((d_lat - 40.7128) * 111.0) ** 2).pow(0.5)
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
    is_jfk_trip = (jfk_p_near + jfk_d_near).clip(0.0, 1.0)
    is_lga_trip = (lga_p_near + lga_d_near).clip(0.0, 1.0)
    is_ewr_trip = (ewr_p_near + ewr_d_near).clip(0.0, 1.0)
    ewr_dist = is_ewr_trip * euclidean
    lga_dist = is_lga_trip * euclidean
    p_man_d_jfk = p_in_manhattan * jfk_d_near
    p_jfk_d_man = jfk_p_near * d_in_manhattan
    p_man_d_ewr = p_in_manhattan * ewr_d_near
    p_ewr_d_man = ewr_p_near * d_in_manhattan
    p_man_d_lga = p_in_manhattan * lga_d_near
    p_lga_d_man = lga_p_near * d_in_manhattan
    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    year_raw = dt.dt.year.astype('float64')
    month_raw = dt.dt.month.astype('float64')
    day_raw = dt.dt.day.astype('float64')
    hour = dt.dt.hour.astype('float64')
    minute = dt.dt.minute.astype('float64')
    exact_hour = hour + minute / 60.0
    dayofweek = dt.dt.dayofweek.astype('float64')
    mta_tax = ((year_raw > 2009.0) | (year_raw == 2009.0) & (month_raw >= 11.0)).astype('float64')
    post_hike = ((year_raw > 2012.0) | (year_raw == 2012.0) & ((month_raw > 9.0) | (month_raw == 9.0) & (day_raw >= 4.0))).astype('float64')
    rot_man_post_hike = rot_manhattan * post_hike
    euclidean_post_hike = euclidean * post_hike
    rot_dx_post_hike = rot_dx * post_hike
    rot_dy_post_hike = rot_dy * post_hike
    intra_man_rot_dist = intra_manhattan * rot_manhattan
    intra_man_post_hike_dist = intra_man_rot_dist * post_hike
    jfk_flat_man = (p_man_d_jfk + p_jfk_d_man).clip(0.0, 1.0)
    jfk_flat_post_hike = jfk_flat_man * post_hike
    jfk_flat_pre_hike = jfk_flat_man * (1.0 - post_hike)
    ewr_flat_man = (p_man_d_ewr + p_ewr_d_man).clip(0.0, 1.0)
    ewr_flat_post_hike = ewr_flat_man * post_hike
    ewr_flat_pre_hike = ewr_flat_man * (1.0 - post_hike)
    ewr_post_hike = is_ewr_trip * post_hike
    ewr_pre_hike = is_ewr_trip * (1.0 - post_hike)
    lga_trip = (p_man_d_lga + p_lga_d_man).clip(0.0, 1.0)
    lga_flat_post_hike = lga_trip * post_hike
    lga_flat_pre_hike = lga_trip * (1.0 - post_hike)
    lga_post_hike = is_lga_trip * post_hike
    lga_pre_hike = is_lga_trip * (1.0 - post_hike)
    is_morning_rush = ((dayofweek < 5.0) & (exact_hour >= 7.0) & (exact_hour < 10.0)).astype('float64')
    is_rush_hour = ((dayofweek < 5.0) & (exact_hour >= 16.0) & (exact_hour < 20.0)).astype('float64')
    is_overnight = ((exact_hour >= 20.0) | (exact_hour < 6.0)).astype('float64')
    is_weekend = (dayofweek >= 5.0).astype('float64')
    jfk_peak = jfk_flat_man * is_rush_hour
    rush_manhattan_dist = intra_manhattan * (is_morning_rush + is_rush_hour).clip(0.0, 1.0) * rot_manhattan
    frac_year = (year_raw + (month_raw - 1.0) / 12.0 + (day_raw - 1.0) / 365.0 - 2009.0) / 6.0
    hour_rad = exact_hour * (2.0 * np.pi / 24.0)
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
    norm_m_lon = (mid_lon - -73.97) * 10.0
    norm_m_lat = (mid_lat - 40.75) * 10.0
    fourier_dict = {}
    coords = {'p_lon': norm_p_lon, 'p_lat': norm_p_lat, 'd_lon': norm_d_lon, 'd_lat': norm_d_lat, 'm_lon': norm_m_lon, 'm_lat': norm_m_lat}
    for c_name, c_val in coords.items():
        for freq in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
            f_arg = c_val * (freq * np.pi)
            f_tag = int(freq)
            fourier_dict[f'{c_name}_sin_{f_tag}'] = f_arg.skb.apply_func(np.sin)
            fourier_dict[f'{c_name}_cos_{f_tag}'] = f_arg.skb.apply_func(np.cos)
    feature_dict = {'pickup_longitude': p_lon, 'pickup_latitude': p_lat, 'dropoff_longitude': d_lon, 'dropoff_latitude': d_lat, 'mid_lon': mid_lon, 'mid_lat': mid_lat, 'mid_rot_x': mid_rot_x, 'mid_rot_y': mid_rot_y, 'dx_km': dx_km, 'dy_km': dy_km, 'abs_dx': abs_dx, 'abs_dy': abs_dy, 'ratio_dx_dy': ratio_dx_dy, 'euclidean': euclidean, 'log_euclidean': log_euclidean, 'cardinal_manhattan': cardinal_manhattan, 'bearing_x': bearing_x, 'bearing_y': bearing_y, 'rot_dx': rot_dx, 'rot_dy': rot_dy, 'rot_manhattan': rot_manhattan, 'log_rot_manhattan': log_rot_manhattan, 'excess_manhattan': excess_manhattan, 'tortuosity': tortuosity, 'is_zero_dist': is_zero_dist, 'p_in_manhattan': p_in_manhattan, 'd_in_manhattan': d_in_manhattan, 'intra_manhattan': intra_manhattan, 'cross_manhattan': cross_manhattan, 'p_in_brooklyn': p_in_brooklyn, 'd_in_brooklyn': d_in_brooklyn, 'p_in_queens': p_in_queens, 'd_in_queens': d_in_queens, 'p_in_bronx': p_in_bronx, 'd_in_bronx': d_in_bronx, 'p_in_si': p_in_si, 'd_in_si': d_in_si, 'si_trip': si_trip, 'cross_si_dist': cross_si_dist, 'dropoff_in_nassau': dropoff_in_nassau, 'nassau_dist': nassau_dist, 'dropoff_in_westchester': dropoff_in_westchester, 'westchester_dist': westchester_dist, 'dropoff_in_nj': dropoff_in_nj, 'nj_dist': nj_dist, 'p_man_d_bk': p_man_d_bk, 'p_bk_d_man': p_bk_d_man, 'p_man_d_qn': p_man_d_qn, 'p_qn_d_man': p_qn_d_man, 'p_man_d_bx': p_man_d_bx, 'p_bx_d_man': p_bx_d_man, 'cross_bk_dist': cross_bk_dist, 'cross_qn_dist': cross_qn_dist, 'cross_bx_dist': cross_bx_dist, 'cross_bk_qn': cross_bk_qn, 'cross_bk_qn_dist': cross_bk_qn_dist, 'midtown_dist_p': midtown_dist_p, 'midtown_dist_d': midtown_dist_d, 'downtown_dist_p': downtown_dist_p, 'downtown_dist_d': downtown_dist_d, 'is_jfk_trip': is_jfk_trip, 'is_lga_trip': is_lga_trip, 'is_ewr_trip': is_ewr_trip, 'ewr_dist': ewr_dist, 'lga_dist': lga_dist, 'p_man_d_jfk': p_man_d_jfk, 'p_jfk_d_man': p_jfk_d_man, 'jfk_flat_post_hike': jfk_flat_post_hike, 'jfk_flat_pre_hike': jfk_flat_pre_hike, 'jfk_peak': jfk_peak, 'p_man_d_ewr': p_man_d_ewr, 'p_ewr_d_man': p_ewr_d_man, 'ewr_flat_post_hike': ewr_flat_post_hike, 'ewr_flat_pre_hike': ewr_flat_pre_hike, 'ewr_post_hike': ewr_post_hike, 'ewr_pre_hike': ewr_pre_hike, 'p_man_d_lga': p_man_d_lga, 'p_lga_d_man': p_lga_d_man, 'lga_flat_post_hike': lga_flat_post_hike, 'lga_flat_pre_hike': lga_flat_pre_hike, 'lga_post_hike': lga_post_hike, 'lga_pre_hike': lga_pre_hike, 'jfk_dist_p': jfk_dist_p, 'jfk_dist_d': jfk_dist_d, 'jfk_p_near': jfk_p_near, 'jfk_d_near': jfk_d_near, 'lga_dist_p': lga_dist_p, 'lga_dist_d': lga_dist_d, 'lga_p_near': lga_p_near, 'lga_d_near': lga_d_near, 'ewr_dist_p': ewr_dist_p, 'ewr_d_dist': ewr_d_dist, 'ewr_p_near': ewr_p_near, 'ewr_d_near': ewr_d_near, 'year': year_raw, 'month': month_raw, 'dayofweek': dayofweek, 'hour': hour, 'exact_hour': exact_hour, 'frac_year': frac_year, 'mta_tax': mta_tax, 'post_hike': post_hike, 'rot_man_post_hike': rot_man_post_hike, 'euclidean_post_hike': euclidean_post_hike, 'rot_dx_post_hike': rot_dx_post_hike, 'rot_dy_post_hike': rot_dy_post_hike, 'intra_man_rot_dist': intra_man_rot_dist, 'intra_man_post_hike_dist': intra_man_post_hike_dist, 'rush_manhattan_dist': rush_manhattan_dist, 'is_morning_rush': is_morning_rush, 'is_rush_hour': is_rush_hour, 'is_overnight': is_overnight, 'is_weekend': is_weekend, 'hour_sin': hour_sin, 'hour_cos': hour_cos, 'dow_sin': dow_sin, 'dow_cos': dow_cos, 'month_sin': month_sin, 'month_cos': month_cos}
    feature_dict.update(fourier_dict)
    features = X[['passenger_count']].astype('float64').assign(**feature_dict).skb.set_name('features')
    model = TriBlendedTaxiRegressor(lgb_weight=0.38, xgb_weight=0.38, torch_weight=0.24, lgb_trees=700, lgb_lr=0.06, lgb_leaves=45, lgb_min_child=50, lgb_reg_lambda=1.0, xgb_trees=850, xgb_lr=0.05, xgb_reg_lambda=2.0, xgb_min_child=20.0, torch_epochs=16, batch_size=2048, lr=0.001, seed=42)
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}