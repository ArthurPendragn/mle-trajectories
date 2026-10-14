from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

class TabularResBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x):
        return x + self.block(x)

class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=256, num_blocks=3, dropout=0.05):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.blocks = nn.ModuleList([TabularResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)

class PyTorchTabularRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, hidden_dim=256, num_blocks=3, dropout=0.05, lr=0.001, weight_decay=0.0001, batch_size=2048, epochs=16, random_state=42):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)
        if hasattr(X, 'values'):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(y, 'values'):
            y_arr = y.values.astype(np.float32)
        else:
            y_arr = np.asarray(y, dtype=np.float32)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.y_mean_ = float(np.mean(y_arr))
        self.y_std_ = float(np.std(y_arr))
        if self.y_std_ < 1e-06:
            self.y_std_ = 1.0
        y_scaled = (y_arr - self.y_mean_) / self.y_std_
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_features = X_arr.shape[1]
        self.net_ = TabularResNet(in_features=in_features, hidden_dim=self.hidden_dim, num_blocks=self.num_blocks, dropout=self.dropout).to(device)
        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-05)
        criterion = nn.MSELoss()
        self.net_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = self.net_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), max_norm=5.0)
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, X):
        self.net_.eval()
        if hasattr(X, 'values'):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        device = next(self.net_.parameters()).device
        dataset = TensorDataset(torch.from_numpy(X_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch_x, in loader:
                batch_x = batch_x.to(device)
                out = self.net_(batch_x)
                preds.append(out.cpu().numpy())
        all_preds = np.concatenate(preds, axis=0)
        all_preds = all_preds * self.y_std_ + self.y_mean_
        return np.clip(all_preds, 2.5, 500.0)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).predict(X)

class EnsembleRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, lgb_weight=0.35, xgb_depth_weight=0.35, xgb_loss_weight=0.2, nn_weight=0.1, random_state=42):
        self.lgb_weight = lgb_weight
        self.xgb_depth_weight = xgb_depth_weight
        self.xgb_loss_weight = xgb_loss_weight
        self.nn_weight = nn_weight
        self.random_state = random_state

    def fit(self, X, y):
        self.lgb_ = LGBMRegressor(n_estimators=750, learning_rate=0.035, num_leaves=255, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=self.random_state, n_jobs=8)
        self.lgb_.fit(X, y)
        self.xgb_depth_ = XGBRegressor(n_estimators=1000, learning_rate=0.04, max_depth=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, random_state=self.random_state, device='cuda', tree_method='hist')
        self.xgb_depth_.fit(X, y)
        self.xgb_loss_ = XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=0, max_leaves=256, grow_policy='lossguide', subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=self.random_state, device='cuda', tree_method='hist')
        self.xgb_loss_.fit(X, y)
        self.nn_ = PyTorchTabularRegressor(hidden_dim=256, num_blocks=3, dropout=0.05, lr=0.001, weight_decay=0.0001, batch_size=2048, epochs=16, random_state=self.random_state)
        self.nn_.fit(X, y)
        return self

    def predict(self, X):
        pred_lgb = self.lgb_.predict(X)
        pred_xgb_depth = self.xgb_depth_.predict(X)
        pred_xgb_loss = self.xgb_loss_.predict(X)
        pred_nn = self.nn_.predict(X)
        blend = self.lgb_weight * pred_lgb + self.xgb_depth_weight * pred_xgb_depth + self.xgb_loss_weight * pred_xgb_loss + self.nn_weight * pred_nn
        return np.clip(blend, 2.5, 500.0)

    def transform(self, X):
        return self.predict(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).predict(X)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    dlon = X['dropoff_longitude'] - X['pickup_longitude']
    dlat = X['dropoff_latitude'] - X['pickup_latitude']
    abs_dlon = dlon.abs()
    abs_dlat = dlat.abs()
    dx = dlon * 82.0
    dy = dlat * 111.0
    euclidean = (dx ** 2 + dy ** 2) ** 0.5
    manhattan = abs_dlon * 82.0 + abs_dlat * 111.0
    theta = -29.0 * (np.pi / 180.0)
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    rot_dx = dx * cos_theta - dy * sin_theta
    rot_dy = dx * sin_theta + dy * cos_theta
    rot_manhattan = rot_dx.abs() + rot_dy.abs()
    tortuosity = rot_manhattan / (euclidean + 0.001)
    p_dx = (X['pickup_longitude'] - -74.0) * 82.0
    p_dy = (X['pickup_latitude'] - 40.7) * 111.0
    pickup_rot_x = p_dx * cos_theta - p_dy * sin_theta
    pickup_rot_y = p_dx * sin_theta + p_dy * cos_theta
    d_dx = (X['dropoff_longitude'] - -74.0) * 82.0
    d_dy = (X['dropoff_latitude'] - 40.7) * 111.0
    dropoff_rot_x = d_dx * cos_theta - d_dy * sin_theta
    dropoff_rot_y = d_dx * sin_theta + d_dy * cos_theta
    center_lon = (X['pickup_longitude'] + X['dropoff_longitude']) * 0.5
    center_lat = (X['pickup_latitude'] + X['dropoff_latitude']) * 0.5
    c_dx = (center_lon - -74.0) * 82.0
    c_dy = (center_lat - 40.7) * 111.0
    center_rot_x = c_dx * cos_theta - c_dy * sin_theta
    center_rot_y = c_dx * sin_theta + c_dy * cos_theta
    is_zero_dist = (euclidean < 0.05).astype('float64')
    dlat_rad = dlat * (np.pi / 180.0)
    dlon_rad = dlon * (np.pi / 180.0)
    lat1_rad = X['pickup_latitude'] * (np.pi / 180.0)
    lat2_rad = X['dropoff_latitude'] * (np.pi / 180.0)
    sin_dlat = (dlat_rad * 0.5).skb.apply_func(np.sin)
    sin_dlon = (dlon_rad * 0.5).skb.apply_func(np.sin)
    cos_lat1 = lat1_rad.skb.apply_func(np.cos)
    cos_lat2 = lat2_rad.skb.apply_func(np.cos)
    a = sin_dlat ** 2 + cos_lat1 * cos_lat2 * sin_dlon ** 2
    haversine = 2.0 * 6371.0 * (a.clip(0.0, 1.0) ** 0.5).skb.apply_func(np.arcsin)
    bearing_cos = dx / (euclidean + 0.001)
    bearing_sin = dy / (euclidean + 0.001)
    jfk_dist_pick = (((X['pickup_longitude'] - -73.7781) * 82.0) ** 2 + ((X['pickup_latitude'] - 40.6413) * 111.0) ** 2) ** 0.5
    jfk_dist_drop = (((X['dropoff_longitude'] - -73.7781) * 82.0) ** 2 + ((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2) ** 0.5
    lga_dist_pick = (((X['pickup_longitude'] - -73.874) * 82.0) ** 2 + ((X['pickup_latitude'] - 40.7769) * 111.0) ** 2) ** 0.5
    lga_dist_drop = (((X['dropoff_longitude'] - -73.874) * 82.0) ** 2 + ((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2) ** 0.5
    ewr_dist_pick = (((X['pickup_longitude'] - -74.1745) * 82.0) ** 2 + ((X['pickup_latitude'] - 40.6895) * 111.0) ** 2) ** 0.5
    ewr_dist_drop = (((X['dropoff_longitude'] - -74.1745) * 82.0) ** 2 + ((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2) ** 0.5
    midtown_dist_pick = (((X['pickup_longitude'] - -73.98) * 82.0) ** 2 + ((X['pickup_latitude'] - 40.755) * 111.0) ** 2) ** 0.5
    midtown_dist_drop = (((X['dropoff_longitude'] - -73.98) * 82.0) ** 2 + ((X['dropoff_latitude'] - 40.755) * 111.0) ** 2) ** 0.5
    is_jfk = ((jfk_dist_pick < 2.5) | (jfk_dist_drop < 2.5)).astype('float64')
    is_lga = ((lga_dist_pick < 2.5) | (lga_dist_drop < 2.5)).astype('float64')
    is_ewr = ((ewr_dist_pick < 2.5) | (ewr_dist_drop < 2.5)).astype('float64')
    pick_in_manhattan = ((X['pickup_longitude'] >= -74.02) & (X['pickup_longitude'] <= -73.93) & (X['pickup_latitude'] >= 40.7) & (X['pickup_latitude'] <= 40.88)).astype('float64')
    drop_in_manhattan = ((X['dropoff_longitude'] >= -74.02) & (X['dropoff_longitude'] <= -73.93) & (X['dropoff_latitude'] >= 40.7) & (X['dropoff_latitude'] <= 40.88)).astype('float64')
    is_jfk_manhattan = ((jfk_dist_pick < 2.5) & ((drop_in_manhattan == 1.0) | (midtown_dist_drop < 5.0)) | (jfk_dist_drop < 2.5) & ((pick_in_manhattan == 1.0) | (midtown_dist_pick < 5.0))).astype('float64')
    is_ewr_manhattan = ((ewr_dist_pick < 2.5) & ((drop_in_manhattan == 1.0) | (midtown_dist_drop < 5.0)) | (ewr_dist_drop < 2.5) & ((pick_in_manhattan == 1.0) | (midtown_dist_pick < 5.0))).astype('float64')
    is_lga_manhattan = ((lga_dist_pick < 2.5) & ((drop_in_manhattan == 1.0) | (midtown_dist_drop < 5.0)) | (lga_dist_drop < 2.5) & ((pick_in_manhattan == 1.0) | (midtown_dist_pick < 5.0))).astype('float64')
    is_both_manhattan = ((pick_in_manhattan == 1.0) & (drop_in_manhattan == 1.0)).astype('float64')
    is_manhattan_cross_borough = ((pick_in_manhattan == 1.0) & (drop_in_manhattan == 0.0) | (pick_in_manhattan == 0.0) & (drop_in_manhattan == 1.0)).astype('float64')
    pick_in_brooklyn = ((X['pickup_longitude'] >= -74.04) & (X['pickup_longitude'] <= -73.85) & (X['pickup_latitude'] >= 40.57) & (X['pickup_latitude'] <= 40.74)).astype('float64')
    drop_in_brooklyn = ((X['dropoff_longitude'] >= -74.04) & (X['dropoff_longitude'] <= -73.85) & (X['dropoff_latitude'] >= 40.57) & (X['dropoff_latitude'] <= 40.74)).astype('float64')
    pick_in_queens = ((X['pickup_longitude'] >= -73.96) & (X['pickup_longitude'] <= -73.7) & (X['pickup_latitude'] >= 40.54) & (X['pickup_latitude'] <= 40.8)).astype('float64')
    drop_in_queens = ((X['dropoff_longitude'] >= -73.96) & (X['dropoff_longitude'] <= -73.7) & (X['dropoff_latitude'] >= 40.54) & (X['dropoff_latitude'] <= 40.8)).astype('float64')
    pick_in_bronx = ((X['pickup_longitude'] >= -73.93) & (X['pickup_longitude'] <= -73.78) & (X['pickup_latitude'] >= 40.79) & (X['pickup_latitude'] <= 40.92)).astype('float64')
    drop_in_bronx = ((X['dropoff_longitude'] >= -73.93) & (X['dropoff_longitude'] <= -73.78) & (X['dropoff_latitude'] >= 40.79) & (X['dropoff_latitude'] <= 40.92)).astype('float64')
    pick_in_staten_island = ((X['pickup_longitude'] >= -74.26) & (X['pickup_longitude'] <= -74.05) & (X['pickup_latitude'] >= 40.49) & (X['pickup_latitude'] <= 40.65)).astype('float64')
    drop_in_staten_island = ((X['dropoff_longitude'] >= -74.26) & (X['dropoff_longitude'] <= -74.05) & (X['dropoff_latitude'] >= 40.49) & (X['dropoff_latitude'] <= 40.65)).astype('float64')
    is_out_of_city = ((X['dropoff_latitude'] > 40.915) | (X['dropoff_longitude'] > -73.68) | (X['dropoff_longitude'] < -74.2) | (X['dropoff_latitude'] < 40.52)).astype('float64')
    dist_out_of_city = rot_manhattan * is_out_of_city
    is_to_new_jersey = ((X['dropoff_longitude'] < -74.04) & (X['dropoff_latitude'] >= 40.65) & (X['dropoff_latitude'] <= 40.88)).astype('float64')
    dist_to_new_jersey = rot_manhattan * is_to_new_jersey
    mid_dist_midtown_tunnel = (((center_lon - -73.9632) * 82.0) ** 2 + ((center_lat - 40.745) * 111.0) ** 2) ** 0.5
    mid_dist_triborough = (((center_lon - -73.9247) * 82.0) ** 2 + ((center_lat - 40.7801) * 111.0) ** 2) ** 0.5
    mid_dist_battery = (((center_lon - -74.0135) * 82.0) ** 2 + ((center_lat - 40.697) * 111.0) ** 2) ** 0.5
    mid_dist_lincoln = (((center_lon - -74.0074) * 82.0) ** 2 + ((center_lat - 40.7614) * 111.0) ** 2) ** 0.5
    mid_dist_holland = (((center_lon - -74.02) * 82.0) ** 2 + ((center_lat - 40.727) * 111.0) ** 2) ** 0.5
    mid_dist_gwb = (((center_lon - -73.9442) * 82.0) ** 2 + ((center_lat - 40.8517) * 111.0) ** 2) ** 0.5
    mid_dist_verrazzano = (((center_lon - -74.0456) * 82.0) ** 2 + ((center_lat - 40.6066) * 111.0) ** 2) ** 0.5
    mid_dist_whitestone = (((center_lon - -73.8306) * 82.0) ** 2 + ((center_lat - 40.8033) * 111.0) ** 2) ** 0.5
    mid_dist_throgs_neck = (((center_lon - -73.7958) * 82.0) ** 2 + ((center_lat - 40.7997) * 111.0) ** 2) ** 0.5
    mid_dist_queensboro = (((center_lon - -73.9537) * 82.0) ** 2 + ((center_lat - 40.757) * 111.0) ** 2) ** 0.5
    mid_dist_brooklyn_bridge = (((center_lon - -73.9969) * 82.0) ** 2 + ((center_lat - 40.7061) * 111.0) ** 2) ** 0.5
    mid_dist_manhattan_bridge = (((center_lon - -73.9906) * 82.0) ** 2 + ((center_lat - 40.7075) * 111.0) ** 2) ** 0.5
    mid_dist_williamsburg = (((center_lon - -73.9692) * 82.0) ** 2 + ((center_lat - 40.7135) * 111.0) ** 2) ** 0.5
    min_dist_toll = mid_dist_midtown_tunnel.skb.apply_func(np.minimum, mid_dist_triborough)
    min_dist_toll = min_dist_toll.skb.apply_func(np.minimum, mid_dist_battery)
    min_dist_toll = min_dist_toll.skb.apply_func(np.minimum, mid_dist_lincoln)
    min_dist_toll = min_dist_toll.skb.apply_func(np.minimum, mid_dist_holland)
    min_dist_toll = min_dist_toll.skb.apply_func(np.minimum, mid_dist_gwb)
    min_dist_toll = min_dist_toll.skb.apply_func(np.minimum, mid_dist_verrazzano)
    min_dist_toll = min_dist_toll.skb.apply_func(np.minimum, mid_dist_whitestone)
    min_dist_toll = min_dist_toll.skb.apply_func(np.minimum, mid_dist_throgs_neck)
    min_dist_free = mid_dist_queensboro.skb.apply_func(np.minimum, mid_dist_brooklyn_bridge)
    min_dist_free = min_dist_free.skb.apply_func(np.minimum, mid_dist_manhattan_bridge)
    min_dist_free = min_dist_free.skb.apply_func(np.minimum, mid_dist_williamsburg)
    toll_vs_free_diff = min_dist_toll - min_dist_free
    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, format='%Y-%m-%d %H:%M:%S UTC', errors='coerce')
    hour = dt.dt.hour.astype('float64').fillna(12.0)
    day = dt.dt.day.astype('float64').fillna(15.0)
    weekday = dt.dt.dayofweek.astype('float64').fillna(3.0)
    month = dt.dt.month.astype('float64').fillna(6.0)
    year = dt.dt.year.astype('float64').fillna(2012.0)
    year_diff = year - 2009.0
    days_since_start = year_diff * 365.25 + (month - 1.0) * 30.4375 + (day - 1.0) + hour / 24.0
    is_post_nov_2009 = ((year > 2009.0) | (year == 2009.0) & (month >= 11.0)).astype('float64')
    is_post_2012_hike = ((year > 2012.0) | (year == 2012.0) & ((month > 9.0) | (month == 9.0) & (day >= 4.0))).astype('float64')
    dist_post_hike = rot_manhattan * is_post_2012_hike
    dist_year = rot_manhattan * year_diff
    jfk_post_hike = is_jfk_manhattan * is_post_2012_hike
    hour_sin = (hour * (2.0 * np.pi / 24.0)).skb.apply_func(np.sin)
    hour_cos = (hour * (2.0 * np.pi / 24.0)).skb.apply_func(np.cos)
    weekday_sin = (weekday * (2.0 * np.pi / 7.0)).skb.apply_func(np.sin)
    weekday_cos = (weekday * (2.0 * np.pi / 7.0)).skb.apply_func(np.cos)
    month_sin = (month * (2.0 * np.pi / 12.0)).skb.apply_func(np.sin)
    month_cos = (month * (2.0 * np.pi / 12.0)).skb.apply_func(np.cos)
    rush_hour = ((weekday < 5.0) & (hour >= 16.0) & (hour < 20.0)).astype('float64')
    overnight = ((hour >= 20.0) | (hour < 6.0)).astype('float64')
    is_weekend = (weekday >= 5.0).astype('float64')
    jfk_rush_hour = is_jfk_manhattan * rush_hour
    jfk_post_hike_rush = is_jfk_manhattan * is_post_2012_hike * rush_hour
    manhattan_to_queens = ((pick_in_manhattan == 1.0) & (drop_in_queens == 1.0) | (pick_in_queens == 1.0) & (drop_in_manhattan == 1.0)).astype('float64')
    manhattan_to_brooklyn = ((pick_in_manhattan == 1.0) & (drop_in_brooklyn == 1.0) | (pick_in_brooklyn == 1.0) & (drop_in_manhattan == 1.0)).astype('float64')
    manhattan_to_bronx = ((pick_in_manhattan == 1.0) & (drop_in_bronx == 1.0) | (pick_in_bronx == 1.0) & (drop_in_manhattan == 1.0)).astype('float64')
    brooklyn_to_queens = ((pick_in_brooklyn == 1.0) & (drop_in_queens == 1.0) | (pick_in_queens == 1.0) & (drop_in_brooklyn == 1.0)).astype('float64')
    is_both_brooklyn = ((pick_in_brooklyn == 1.0) & (drop_in_brooklyn == 1.0)).astype('float64')
    is_both_queens = ((pick_in_queens == 1.0) & (drop_in_queens == 1.0)).astype('float64')
    log_rot_manhattan = (rot_manhattan + 0.1).skb.apply_func(np.log)
    log_euclidean = (euclidean + 0.1).skb.apply_func(np.log)
    dist_rush_hour = rot_manhattan * rush_hour
    dist_overnight = rot_manhattan * overnight
    dist_weekend = rot_manhattan * is_weekend
    dist_intra_manhattan = rot_manhattan * is_both_manhattan
    is_manhattan_rush_hour = is_both_manhattan * rush_hour
    dist_manhattan_rush_hour = rot_manhattan * is_manhattan_rush_hour
    geo_temporal_features = X.assign(dlon=dlon, dlat=dlat, abs_dlon=abs_dlon, abs_dlat=abs_dlat, euclidean=euclidean, manhattan=manhattan, rot_dx=rot_dx, rot_dy=rot_dy, rot_manhattan=rot_manhattan, tortuosity=tortuosity, pickup_rot_x=pickup_rot_x, pickup_rot_y=pickup_rot_y, dropoff_rot_x=dropoff_rot_x, dropoff_rot_y=dropoff_rot_y, center_lon=center_lon, center_lat=center_lat, center_rot_x=center_rot_x, center_rot_y=center_rot_y, is_zero_dist=is_zero_dist, haversine=haversine, bearing_sin=bearing_sin, bearing_cos=bearing_cos, jfk_dist_pick=jfk_dist_pick, jfk_dist_drop=jfk_dist_drop, lga_dist_pick=lga_dist_pick, lga_dist_drop=lga_dist_drop, ewr_dist_pick=ewr_dist_pick, ewr_dist_drop=ewr_dist_drop, midtown_dist_pick=midtown_dist_pick, midtown_dist_drop=midtown_dist_drop, is_jfk=is_jfk, is_lga=is_lga, is_ewr=is_ewr, is_jfk_manhattan=is_jfk_manhattan, is_ewr_manhattan=is_ewr_manhattan, is_lga_manhattan=is_lga_manhattan, pick_in_manhattan=pick_in_manhattan, drop_in_manhattan=drop_in_manhattan, is_both_manhattan=is_both_manhattan, is_manhattan_cross_borough=is_manhattan_cross_borough, pick_in_brooklyn=pick_in_brooklyn, drop_in_brooklyn=drop_in_brooklyn, pick_in_queens=pick_in_queens, drop_in_queens=drop_in_queens, pick_in_bronx=pick_in_bronx, drop_in_bronx=drop_in_bronx, pick_in_staten_island=pick_in_staten_island, drop_in_staten_island=drop_in_staten_island, is_out_of_city=is_out_of_city, dist_out_of_city=dist_out_of_city, is_to_new_jersey=is_to_new_jersey, dist_to_new_jersey=dist_to_new_jersey, mid_dist_midtown_tunnel=mid_dist_midtown_tunnel, mid_dist_triborough=mid_dist_triborough, mid_dist_battery=mid_dist_battery, mid_dist_lincoln=mid_dist_lincoln, mid_dist_holland=mid_dist_holland, mid_dist_gwb=mid_dist_gwb, mid_dist_verrazzano=mid_dist_verrazzano, mid_dist_whitestone=mid_dist_whitestone, mid_dist_throgs_neck=mid_dist_throgs_neck, mid_dist_queensboro=mid_dist_queensboro, mid_dist_brooklyn_bridge=mid_dist_brooklyn_bridge, mid_dist_manhattan_bridge=mid_dist_manhattan_bridge, mid_dist_williamsburg=mid_dist_williamsburg, min_dist_toll=min_dist_toll, min_dist_free=min_dist_free, toll_vs_free_diff=toll_vs_free_diff, hour=hour, weekday=weekday, month=month, year_diff=year_diff, days_since_start=days_since_start, is_post_nov_2009=is_post_nov_2009, is_post_2012_hike=is_post_2012_hike, dist_post_hike=dist_post_hike, dist_year=dist_year, jfk_post_hike=jfk_post_hike, jfk_rush_hour=jfk_rush_hour, jfk_post_hike_rush=jfk_post_hike_rush, manhattan_to_queens=manhattan_to_queens, manhattan_to_brooklyn=manhattan_to_brooklyn, manhattan_to_bronx=manhattan_to_bronx, brooklyn_to_queens=brooklyn_to_queens, is_both_brooklyn=is_both_brooklyn, is_both_queens=is_both_queens, log_rot_manhattan=log_rot_manhattan, log_euclidean=log_euclidean, dist_rush_hour=dist_rush_hour, dist_overnight=dist_overnight, is_weekend=is_weekend, dist_weekend=dist_weekend, dist_intra_manhattan=dist_intra_manhattan, is_manhattan_rush_hour=is_manhattan_rush_hour, dist_manhattan_rush_hour=dist_manhattan_rush_hour, hour_sin=hour_sin, hour_cos=hour_cos, weekday_sin=weekday_sin, weekday_cos=weekday_cos, month_sin=month_sin, month_cos=month_cos, rush_hour=rush_hour, overnight=overnight)[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'dlon', 'dlat', 'abs_dlon', 'abs_dlat', 'euclidean', 'manhattan', 'rot_dx', 'rot_dy', 'rot_manhattan', 'tortuosity', 'pickup_rot_x', 'pickup_rot_y', 'dropoff_rot_x', 'dropoff_rot_y', 'center_lon', 'center_lat', 'center_rot_x', 'center_rot_y', 'is_zero_dist', 'haversine', 'bearing_sin', 'bearing_cos', 'jfk_dist_pick', 'jfk_dist_drop', 'lga_dist_pick', 'lga_dist_drop', 'ewr_dist_pick', 'ewr_dist_drop', 'midtown_dist_pick', 'midtown_dist_drop', 'is_jfk', 'is_lga', 'is_ewr', 'is_jfk_manhattan', 'is_ewr_manhattan', 'is_lga_manhattan', 'pick_in_manhattan', 'drop_in_manhattan', 'is_both_manhattan', 'is_manhattan_cross_borough', 'pick_in_brooklyn', 'drop_in_brooklyn', 'pick_in_queens', 'drop_in_queens', 'pick_in_bronx', 'drop_in_bronx', 'pick_in_staten_island', 'drop_in_staten_island', 'is_out_of_city', 'dist_out_of_city', 'is_to_new_jersey', 'dist_to_new_jersey', 'mid_dist_midtown_tunnel', 'mid_dist_triborough', 'mid_dist_battery', 'mid_dist_lincoln', 'mid_dist_holland', 'mid_dist_gwb', 'mid_dist_verrazzano', 'mid_dist_whitestone', 'mid_dist_throgs_neck', 'mid_dist_queensboro', 'mid_dist_brooklyn_bridge', 'mid_dist_manhattan_bridge', 'mid_dist_williamsburg', 'min_dist_toll', 'min_dist_free', 'toll_vs_free_diff', 'hour', 'weekday', 'month', 'year_diff', 'days_since_start', 'is_post_nov_2009', 'is_post_2012_hike', 'dist_post_hike', 'dist_year', 'jfk_post_hike', 'jfk_rush_hour', 'jfk_post_hike_rush', 'manhattan_to_queens', 'manhattan_to_brooklyn', 'manhattan_to_bronx', 'brooklyn_to_queens', 'is_both_brooklyn', 'is_both_queens', 'log_rot_manhattan', 'log_euclidean', 'dist_rush_hour', 'dist_overnight', 'is_weekend', 'dist_weekend', 'dist_intra_manhattan', 'is_manhattan_rush_hour', 'dist_manhattan_rush_hour', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 'rush_hour', 'overnight']].skb.set_name('geo_temporal_features')
    scaled_features = geo_temporal_features.skb.apply(StandardScaler()).skb.set_name('scaled_features')
    model = EnsembleRegressor(lgb_weight=0.35, xgb_depth_weight=0.35, xgb_loss_weight=0.2, nn_weight=0.1, random_state=42)
    pred = scaled_features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}