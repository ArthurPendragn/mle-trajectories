import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

class ResBlock(nn.Module):

    def __init__(self, dim: int, dropout: float=0.1):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class WideAndDeepTabularResNet(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int=256, num_blocks: int=3, dropout: float=0.1):
        super().__init__()
        self.wide = nn.Linear(in_features, 1, bias=False)
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
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

    def __init__(self, hidden_dim: int=256, num_blocks: int=3, dropout: float=0.1, lr: float=0.003, epochs: int=25, batch_size: int=2048, random_state: int=42):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
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
        net = WideAndDeepTabularResNet(in_features=in_dim, hidden_dim=self.hidden_dim, num_blocks=self.num_blocks, dropout=self.dropout).to(device)
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

class StackingResidualRegressor(RegressorMixin, BaseEstimator):

    def __init__(self, hidden_dim: int=256, num_blocks: int=3, dropout: float=0.1, lr: float=0.003, epochs: int=25, batch_size: int=2048, lgb_n_estimators: int=1000, lgb_learning_rate: float=0.04, lgb_num_leaves: int=63, lgb_min_child_samples: int=50, lgb_reg_alpha: float=1.0, lgb_reg_lambda: float=5.0, lgb_colsample_bytree: float=0.8, lgb_subsample: float=0.8, lgb_subsample_freq: int=1, xgb_n_estimators: int=1000, xgb_learning_rate: float=0.04, xgb_max_depth: int=8, xgb_reg_alpha: float=1.0, xgb_reg_lambda: float=5.0, xgb_colsample_bytree: float=0.8, xgb_subsample: float=0.8, val_size: float=0.15, fare_floor: float=2.5, random_state: int=42, n_jobs: int=8):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lgb_n_estimators = lgb_n_estimators
        self.lgb_learning_rate = lgb_learning_rate
        self.lgb_num_leaves = lgb_num_leaves
        self.lgb_min_child_samples = lgb_min_child_samples
        self.lgb_reg_alpha = lgb_reg_alpha
        self.lgb_reg_lambda = lgb_reg_lambda
        self.lgb_colsample_bytree = lgb_colsample_bytree
        self.lgb_subsample = lgb_subsample
        self.lgb_subsample_freq = lgb_subsample_freq
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_learning_rate = xgb_learning_rate
        self.xgb_max_depth = xgb_max_depth
        self.xgb_reg_alpha = xgb_reg_alpha
        self.xgb_reg_lambda = xgb_reg_lambda
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_subsample = xgb_subsample
        self.val_size = val_size
        self.fare_floor = fare_floor
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.baseline_col_ = 'statutory_fare_estimate'
            statutory_baseline = X['statutory_fare_estimate'].to_numpy(dtype=np.float32)
            self.baseline_idx_ = X.columns.get_loc('statutory_fare_estimate')
        else:
            self.baseline_idx_ = -1
            statutory_baseline = X[:, self.baseline_idx_].astype(np.float32)
        X_mat = np.asarray(X, dtype=np.float32)
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        residual = y_arr - statutory_baseline

        n_samples = len(y_arr)
        rng = np.random.RandomState(self.random_state)
        val_mask = rng.rand(n_samples) < self.val_size
        train_mask = ~val_mask

        X_tr = X.iloc[train_mask] if isinstance(X, pd.DataFrame) else X_mat[train_mask]
        y_res_tr = residual[train_mask]
        X_mat_tr = X_mat[train_mask]

        X_val = X.iloc[val_mask] if isinstance(X, pd.DataFrame) else X_mat[val_mask]
        y_res_val = residual[val_mask]
        X_mat_val = X_mat[val_mask]

        self.resnet_ = TabularResNetRegressor(hidden_dim=self.hidden_dim, num_blocks=self.num_blocks, dropout=self.dropout, lr=self.lr, epochs=self.epochs, batch_size=self.batch_size, random_state=self.random_state)
        self.resnet_.fit(X_mat_tr, y_res_tr)

        self.lgb_ = LGBMRegressor(n_estimators=self.lgb_n_estimators, learning_rate=self.lgb_learning_rate, num_leaves=self.lgb_num_leaves, min_child_samples=self.lgb_min_child_samples, reg_alpha=self.lgb_reg_alpha, reg_lambda=self.lgb_reg_lambda, colsample_bytree=self.lgb_colsample_bytree, subsample=self.lgb_subsample, subsample_freq=self.lgb_subsample_freq, random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)
        self.lgb_.fit(X_tr, y_res_tr)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.xgb_ = XGBRegressor(n_estimators=self.xgb_n_estimators, learning_rate=self.xgb_learning_rate, max_depth=self.xgb_max_depth, reg_alpha=self.xgb_reg_alpha, reg_lambda=self.xgb_reg_lambda, colsample_bytree=self.xgb_colsample_bytree, subsample=self.xgb_subsample, random_state=self.random_state, n_jobs=self.n_jobs, tree_method='hist', device=device)
            self.xgb_.fit(X_mat_tr, y_res_tr)
        except Exception:
            self.xgb_ = XGBRegressor(n_estimators=self.xgb_n_estimators, learning_rate=self.xgb_learning_rate, max_depth=self.xgb_max_depth, reg_alpha=self.xgb_reg_alpha, reg_lambda=self.xgb_reg_lambda, colsample_bytree=self.xgb_colsample_bytree, subsample=self.xgb_subsample, random_state=self.random_state, n_jobs=self.n_jobs, tree_method='hist')
            self.xgb_.fit(X_mat_tr, y_res_tr)

        val_p_resnet = self.resnet_.predict(X_mat_val)
        val_p_lgb = self.lgb_.predict(X_val)
        val_p_xgb = self.xgb_.predict(X_mat_val)
        S_val = np.column_stack([val_p_resnet, val_p_lgb, val_p_xgb])

        self.meta_model_ = LinearRegression(positive=True)
        self.meta_model_.fit(S_val, y_res_val)

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            statutory_baseline = X['statutory_fare_estimate'].to_numpy(dtype=np.float32)
        else:
            statutory_baseline = X[:, self.baseline_idx_].astype(np.float32)
        X_mat = np.asarray(X, dtype=np.float32)
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

        p_resnet = self.resnet_.predict(X_mat)
        p_lgb = self.lgb_.predict(X)
        p_xgb = self.xgb_.predict(X_mat)
        S_test = np.column_stack([p_resnet, p_lgb, p_xgb])

        pred_res = self.meta_model_.predict(S_test)
        pred = statutory_baseline + pred_res
        if self.fare_floor is not None:
            pred = np.clip(pred, self.fare_floor, None)
        return pred

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

    pick_x_km = (X['pickup_longitude'] - -74.0) * 84.18
    pick_y_km = (X['pickup_latitude'] - 40.7) * 111.13
    drop_x_km = (X['dropoff_longitude'] - -74.0) * 84.18
    drop_y_km = (X['dropoff_latitude'] - 40.7) * 111.13
    rot_pick_x = pick_x_km * cos_t - pick_y_km * sin_t
    rot_pick_y = pick_x_km * sin_t + pick_y_km * cos_t
    rot_drop_x = drop_x_km * cos_t - drop_y_km * sin_t
    rot_drop_y = drop_x_km * sin_t + drop_y_km * cos_t

    rad_conv = np.pi / 180.0
    lat1_rad = X['pickup_latitude'] * rad_conv
    lat2_rad = X['dropoff_latitude'] * rad_conv
    dlat_rad = raw_d_lat * rad_conv
    dlon_rad = raw_d_lon * rad_conv
    a_hav = (dlat_rad / 2.0).skb.apply_func(np.sin) ** 2 + lat1_rad.skb.apply_func(np.cos) * lat2_rad.skb.apply_func(np.cos) * (dlon_rad / 2.0).skb.apply_func(np.sin) ** 2
    haversine_km = 6371.0 * 2.0 * a_hav.skb.apply_func(np.arcsin)

    tortuosity = rotated_manhattan_km / (distance_km + 0.001)
    log_distance_km = (distance_km + 1.0).skb.apply_func(np.log)
    log_rotated_manhattan_km = (rotated_manhattan_km + 1.0).skb.apply_func(np.log)

    sin_bearing = raw_dx_km / (distance_km + 1e-05)
    cos_bearing = raw_dy_km / (distance_km + 1e-05)

    is_zero_dist = (distance_km < 0.05).astype(float)
    crosstown_ratio = rot_dx / (rotated_manhattan_km + 0.001)

    jfk_drop_dist = (((X['dropoff_latitude'] - 40.6445) * 111.13) ** 2 + ((X['dropoff_longitude'] - -73.7825) * 84.18) ** 2) ** 0.5
    lga_drop_dist = (((X['dropoff_latitude'] - 40.7735) * 111.13) ** 2 + ((X['dropoff_longitude'] - -73.872) * 84.18) ** 2) ** 0.5
    ewr_drop_dist = (((X['dropoff_latitude'] - 40.6895) * 111.13) ** 2 + ((X['dropoff_longitude'] - -74.1745) * 84.18) ** 2) ** 0.5
    jfk_pick_dist = (((X['pickup_latitude'] - 40.6445) * 111.13) ** 2 + ((X['pickup_longitude'] - -73.7825) * 84.18) ** 2) ** 0.5
    lga_pick_dist = (((X['pickup_latitude'] - 40.7735) * 111.13) ** 2 + ((X['pickup_longitude'] - -73.872) * 84.18) ** 2) ** 0.5
    ewr_pick_dist = (((X['pickup_latitude'] - 40.6895) * 111.13) ** 2 + ((X['pickup_longitude'] - -74.1745) * 84.18) ** 2) ** 0.5
    is_jfk = ((jfk_drop_dist < 2.2) | (jfk_pick_dist < 2.2)).astype(float)
    is_lga = ((lga_drop_dist < 2.0) | (lga_pick_dist < 2.0)).astype(float)
    is_ewr = ((ewr_drop_dist < 2.2) | (ewr_pick_dist < 2.2)).astype(float)

    center_lat = (X['pickup_latitude'] + X['dropoff_latitude']) / 2.0
    center_lon = (X['pickup_longitude'] + X['dropoff_longitude']) / 2.0
    midtown_pick_dist = (((X['pickup_latitude'] - 40.755) * 111.13) ** 2 + ((X['pickup_longitude'] - -73.985) * 84.18) ** 2) ** 0.5
    midtown_drop_dist = (((X['dropoff_latitude'] - 40.755) * 111.13) ** 2 + ((X['dropoff_longitude'] - -73.985) * 84.18) ** 2) ** 0.5
    center_dist_midtown = (((center_lat - 40.755) * 111.13) ** 2 + ((center_lon - -73.985) * 84.18) ** 2) ** 0.5
    fidi_pick_dist = (((X['pickup_latitude'] - 40.7128) * 111.13) ** 2 + ((X['pickup_longitude'] - -74.006) * 84.18) ** 2) ** 0.5
    fidi_drop_dist = (((X['dropoff_latitude'] - 40.7128) * 111.13) ** 2 + ((X['dropoff_longitude'] - -74.006) * 84.18) ** 2) ** 0.5

    mid_midtown_tunnel = (((center_lat - 40.744) * 111.13) ** 2 + ((center_lon - -73.968) * 84.18) ** 2) ** 0.5
    mid_rfk = (((center_lat - 40.78) * 111.13) ** 2 + ((center_lon - -73.924) * 84.18) ** 2) ** 0.5
    mid_queensboro = (((center_lat - 40.757) * 111.13) ** 2 + ((center_lon - -73.955) * 84.18) ** 2) ** 0.5
    mid_battery_tunnel = (((center_lat - 40.697) * 111.13) ** 2 + ((center_lon - -74.015) * 84.18) ** 2) ** 0.5
    mid_brooklyn_bridge = (((center_lat - 40.706) * 111.13) ** 2 + ((center_lon - -73.997) * 84.18) ** 2) ** 0.5
    mid_lincoln = (((center_lat - 40.76) * 111.13) ** 2 + ((center_lon - -74.004) * 84.18) ** 2) ** 0.5
    mid_holland = (((center_lat - 40.726) * 111.13) ** 2 + ((center_lon - -74.016) * 84.18) ** 2) ** 0.5
    mid_gwb = (((center_lat - 40.852) * 111.13) ** 2 + ((center_lon - -73.953) * 84.18) ** 2) ** 0.5

    is_pick_manhattan = ((X['pickup_longitude'] >= -74.02) & (X['pickup_longitude'] <= -73.93) & (X['pickup_latitude'] >= 40.7) & (X['pickup_latitude'] <= 40.87)).fillna(False)
    is_drop_manhattan = ((X['dropoff_longitude'] >= -74.02) & (X['dropoff_longitude'] <= -73.93) & (X['dropoff_latitude'] >= 40.7) & (X['dropoff_latitude'] <= 40.87)).fillna(False)
    is_pick_brooklyn = ((X['pickup_longitude'] >= -74.04) & (X['pickup_longitude'] <= -73.85) & (X['pickup_latitude'] >= 40.57) & (X['pickup_latitude'] <= 40.73)).fillna(False)
    is_drop_brooklyn = ((X['dropoff_longitude'] >= -74.04) & (X['dropoff_longitude'] <= -73.85) & (X['dropoff_latitude'] >= 40.57) & (X['dropoff_latitude'] <= 40.73)).fillna(False)
    is_pick_queens = ((X['pickup_longitude'] >= -73.96) & (X['pickup_longitude'] <= -73.7) & (X['pickup_latitude'] >= 40.54) & (X['pickup_latitude'] <= 40.8)).fillna(False)
    is_drop_queens = ((X['dropoff_longitude'] >= -73.96) & (X['dropoff_longitude'] <= -73.7) & (X['dropoff_latitude'] >= 40.54) & (X['dropoff_latitude'] <= 40.8)).fillna(False)
    is_pick_bronx = ((X['pickup_longitude'] >= -73.93) & (X['pickup_longitude'] <= -73.76) & (X['pickup_latitude'] >= 40.8) & (X['pickup_latitude'] <= 40.91)).fillna(False)
    is_drop_bronx = ((X['dropoff_longitude'] >= -73.93) & (X['dropoff_longitude'] <= -73.76) & (X['dropoff_latitude'] >= 40.8) & (X['dropoff_latitude'] <= 40.91)).fillna(False)

    is_suburban_dropoff = ((X['dropoff_latitude'] > 40.915) | (X['dropoff_longitude'] > -73.7)).astype(float)
    is_suburban_pickup = ((X['pickup_latitude'] > 40.915) | (X['pickup_longitude'] > -73.7)).astype(float)

    is_both_manhattan = (is_pick_manhattan & is_drop_manhattan).astype(float)
    is_cross_borough = (is_pick_manhattan != is_drop_manhattan).astype(float)
    is_manhattan_brooklyn = (is_pick_manhattan & is_drop_brooklyn | is_drop_manhattan & is_pick_brooklyn).astype(float)
    is_manhattan_queens = (is_pick_manhattan & is_drop_queens | is_drop_manhattan & is_pick_queens).astype(float)
    is_manhattan_bronx = (is_pick_manhattan & is_drop_bronx | is_drop_manhattan & is_pick_bronx).astype(float)

    is_jfk_manhattan = ((jfk_pick_dist < 2.2) & is_drop_manhattan | (jfk_drop_dist < 2.2) & is_pick_manhattan).astype(float)
    is_lga_manhattan = ((lga_pick_dist < 2.0) & is_drop_manhattan | (lga_drop_dist < 2.0) & is_pick_manhattan).astype(float)
    is_ewr_manhattan = ((ewr_pick_dist < 2.2) & is_drop_manhattan | (ewr_drop_dist < 2.2) & is_pick_manhattan).astype(float)

    tolled_midtown_tunnel = ((is_manhattan_queens > 0) & (mid_midtown_tunnel < 3.0)).astype(float)
    tolled_rfk = (((is_manhattan_queens > 0) | (is_manhattan_bronx > 0)) & (mid_rfk < 3.0)).astype(float)
    tolled_battery_tunnel = ((is_manhattan_brooklyn > 0) & (mid_battery_tunnel < 2.5)).astype(float)
    tolled_hudson_tunnel = ((is_ewr_manhattan > 0) | (X['dropoff_longitude'] < -74.03)).astype(float)

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek
    hour = dt.dt.hour
    year_fraction = year + (month - 1) / 12.0 + (day - 1) / 365.25

    hour_rad = hour * (2.0 * np.pi / 24.0)
    sin_hour = hour_rad.skb.apply_func(np.sin)
    cos_hour = hour_rad.skb.apply_func(np.cos)

    dow_rad = dayofweek * (2.0 * np.pi / 7.0)
    sin_dow = dow_rad.skb.apply_func(np.sin)
    cos_dow = dow_rad.skb.apply_func(np.cos)

    is_post_nov_2009 = ((year > 2009) | (year == 2009) & (month >= 11)).fillna(False).astype(float)
    is_post_2012 = ((year > 2012) | (year == 2012) & ((month > 9) | (month == 9) & (day >= 4))).fillna(False).astype(float)

    dist_x_post_2012 = distance_km * is_post_2012
    rot_dist_x_post_2012 = rotated_manhattan_km * is_post_2012
    jfk_x_post_2012 = is_jfk * is_post_2012
    jfk_manhattan_x_post_2012 = is_jfk_manhattan * is_post_2012

    is_rush_hour = ((dayofweek < 5) & (hour >= 16) & (hour < 20)).fillna(False).astype(float)
    is_morning_rush = ((dayofweek < 5) & (hour >= 7) & (hour < 10)).fillna(False).astype(float)
    is_weekday_midday = ((dayofweek < 5) & (hour >= 10) & (hour < 16)).fillna(False).astype(float)
    is_weekend_night = ((dayofweek >= 4) & ((hour >= 22) | (hour < 4))).fillna(False).astype(float)
    is_overnight = ((hour >= 20) | (hour < 6)).fillna(False).astype(float)

    statutory_surcharge = 0.5 * is_post_nov_2009 + 0.5 * is_overnight + 1.0 * is_rush_hour
    ewr_surcharge = is_ewr_manhattan * (15.0 * (1.0 - is_post_2012) + 17.5 * is_post_2012)
    rate_per_km = 1.2427 * (1.0 - is_post_2012) + 1.5534 * is_post_2012
    effective_dist_km = is_both_manhattan * rotated_manhattan_km + (1.0 - is_both_manhattan) * (0.6 * rotated_manhattan_km + 0.4 * haversine_km)
    metered_dist_fare = effective_dist_km * rate_per_km

    jfk_flat_rate = is_jfk_manhattan * (45.0 * (1.0 - is_post_2012) + 52.0 * is_post_2012)
    base_fare = 2.5 + metered_dist_fare
    statutory_fare_estimate = (1.0 - is_jfk_manhattan) * base_fare + jfk_flat_rate + statutory_surcharge + ewr_surcharge

    features = coords.assign(
        d_lat=d_lat,
        d_lon=d_lon,
        distance_km=distance_km,
        manhattan_km=manhattan_km,
        rot_dx=rot_dx,
        rot_dy=rot_dy,
        rotated_manhattan_km=rotated_manhattan_km,
        rot_pick_x=rot_pick_x,
        rot_pick_y=rot_pick_y,
        rot_drop_x=rot_drop_x,
        rot_drop_y=rot_drop_y,
        haversine_km=haversine_km,
        tortuosity=tortuosity,
        log_distance_km=log_distance_km,
        log_rotated_manhattan_km=log_rotated_manhattan_km,
        sin_bearing=sin_bearing,
        cos_bearing=cos_bearing,
        is_zero_dist=is_zero_dist,
        crosstown_ratio=crosstown_ratio,
        jfk_drop_dist=jfk_drop_dist,
        lga_drop_dist=lga_drop_dist,
        ewr_drop_dist=ewr_drop_dist,
        jfk_pick_dist=jfk_pick_dist,
        lga_pick_dist=lga_pick_dist,
        ewr_pick_dist=ewr_pick_dist,
        is_jfk=is_jfk,
        is_lga=is_lga,
        is_ewr=is_ewr,
        midtown_pick_dist=midtown_pick_dist,
        midtown_drop_dist=midtown_drop_dist,
        center_dist_midtown=center_dist_midtown,
        fidi_pick_dist=fidi_pick_dist,
        fidi_drop_dist=fidi_drop_dist,
        mid_midtown_tunnel=mid_midtown_tunnel,
        mid_rfk=mid_rfk,
        mid_queensboro=mid_queensboro,
        mid_battery_tunnel=mid_battery_tunnel,
        mid_brooklyn_bridge=mid_brooklyn_bridge,
        mid_lincoln=mid_lincoln,
        mid_holland=mid_holland,
        mid_gwb=mid_gwb,
        tolled_midtown_tunnel=tolled_midtown_tunnel,
        tolled_rfk=tolled_rfk,
        tolled_battery_tunnel=tolled_battery_tunnel,
        tolled_hudson_tunnel=tolled_hudson_tunnel,
        is_suburban_dropoff=is_suburban_dropoff,
        is_suburban_pickup=is_suburban_pickup,
        is_both_manhattan=is_both_manhattan,
        is_cross_borough=is_cross_borough,
        is_manhattan_brooklyn=is_manhattan_brooklyn,
        is_manhattan_queens=is_manhattan_queens,
        is_manhattan_bronx=is_manhattan_bronx,
        is_jfk_manhattan=is_jfk_manhattan,
        is_lga_manhattan=is_lga_manhattan,
        is_ewr_manhattan=is_ewr_manhattan,
        jfk_manhattan_x_post_2012=jfk_manhattan_x_post_2012,
        year=year,
        month=month,
        day=day,
        dayofweek=dayofweek,
        hour=hour,
        year_fraction=year_fraction,
        sin_hour=sin_hour,
        cos_hour=cos_hour,
        sin_dow=sin_dow,
        cos_dow=cos_dow,
        is_post_nov_2009=is_post_nov_2009,
        is_post_2012=is_post_2012,
        dist_x_post_2012=dist_x_post_2012,
        rot_dist_x_post_2012=rot_dist_x_post_2012,
        jfk_x_post_2012=jfk_x_post_2012,
        is_rush_hour=is_rush_hour,
        is_morning_rush=is_morning_rush,
        is_weekday_midday=is_weekday_midday,
        is_weekend_night=is_weekend_night,
        is_overnight=is_overnight,
        statutory_surcharge=statutory_surcharge,
        ewr_surcharge=ewr_surcharge,
        statutory_fare_estimate=statutory_fare_estimate
    ).skb.set_name('features')

    model = StackingResidualRegressor(
        hidden_dim=256,
        num_blocks=3,
        dropout=0.1,
        lr=0.003,
        epochs=25,
        batch_size=2048,
        lgb_n_estimators=1000,
        lgb_learning_rate=0.04,
        lgb_num_leaves=63,
        lgb_min_child_samples=50,
        lgb_reg_alpha=1.0,
        lgb_reg_lambda=5.0,
        lgb_colsample_bytree=0.8,
        lgb_subsample=0.8,
        lgb_subsample_freq=1,
        xgb_n_estimators=1000,
        xgb_learning_rate=0.04,
        xgb_max_depth=8,
        xgb_reg_alpha=1.0,
        xgb_reg_lambda=5.0,
        xgb_colsample_bytree=0.8,
        xgb_subsample=0.8,
        val_size=0.15,
        fare_floor=2.5,
        random_state=42,
        n_jobs=8
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': scoring}