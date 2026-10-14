import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import skrub

class TabularPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, n_fourier_octaves=6):
        self.n_fourier_octaves = n_fourier_octaves

    def _extract_raw(self, X):
        df = pd.DataFrame(index=X.index)
        p_lon = X['pickup_longitude'].astype('float64')
        p_lat = X['pickup_latitude'].astype('float64')
        d_lon = X['dropoff_longitude'].astype('float64')
        d_lat = X['dropoff_latitude'].astype('float64')
        df['p_lon'] = p_lon
        df['p_lat'] = p_lat
        df['d_lon'] = d_lon
        df['d_lat'] = d_lat
        d_lat_diff = d_lat - p_lat
        d_lon_diff = d_lon - p_lon
        df['d_lat_diff'] = d_lat_diff
        df['d_lon_diff'] = d_lon_diff
        df['d_lat_abs'] = d_lat_diff.abs()
        df['d_lon_abs'] = d_lon_diff.abs()
        df['mid_lon'] = 0.5 * (p_lon + d_lon)
        df['mid_lat'] = 0.5 * (p_lat + d_lat)
        angle_rad = np.radians(29.0)
        cos_ang = np.cos(angle_rad)
        sin_ang = np.sin(angle_rad)
        p_rot_x = cos_ang * p_lon - sin_ang * p_lat
        p_rot_y = sin_ang * p_lon + cos_ang * p_lat
        d_rot_x = cos_ang * d_lon - sin_ang * d_lat
        d_rot_y = sin_ang * d_lon + cos_ang * d_lat
        df['p_rot_x'] = p_rot_x
        df['p_rot_y'] = p_rot_y
        df['d_rot_x'] = d_rot_x
        df['d_rot_y'] = d_rot_y
        rot_dist_x = (d_rot_x - p_rot_x).abs()
        rot_dist_y = (d_rot_y - p_rot_y).abs()
        df['rot_dist_x'] = rot_dist_x
        df['rot_dist_y'] = rot_dist_y
        rot_manhattan = rot_dist_x + rot_dist_y
        df['rot_manhattan'] = rot_manhattan
        df['euclidean'] = np.sqrt(df['d_lat_abs'] ** 2 + df['d_lon_abs'] ** 2)
        lat1 = np.radians(p_lat)
        lat2 = np.radians(d_lat)
        dlat = np.radians(d_lat_diff)
        dlon = np.radians(d_lon_diff)
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        haversine_km = 6371.0 * c
        df['haversine_km'] = haversine_km
        manhattan_km = 111.0 * df['d_lat_abs'] + 111.0 * np.cos(lat1) * df['d_lon_abs']
        df['manhattan_km'] = manhattan_km
        y_bear = np.sin(dlon) * np.cos(lat2)
        x_bear = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        df['bearing'] = np.arctan2(y_bear, x_bear)
        landmarks = {'jfk': (-73.7781, 40.6413), 'lga': (-73.874, 40.7769), 'ewr': (-74.1745, 40.6895), 'midtown': (-73.9851, 40.758)}
        for name, (l_lon, l_lat) in landmarks.items():
            l_lat_rad = np.radians(l_lat)
            dlat_p = np.radians(l_lat - p_lat)
            dlon_p = np.radians(l_lon - p_lon)
            ap = np.sin(dlat_p / 2.0) ** 2 + np.cos(lat1) * np.cos(l_lat_rad) * np.sin(dlon_p / 2.0) ** 2
            p_dist = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(ap, 0.0, 1.0)))
            df[f'p_dist_{name}'] = p_dist
            dlat_d = np.radians(l_lat - d_lat)
            dlon_d = np.radians(l_lon - d_lon)
            ad = np.sin(dlat_d / 2.0) ** 2 + np.cos(lat2) * np.cos(l_lat_rad) * np.sin(dlon_d / 2.0) ** 2
            d_dist = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(ad, 0.0, 1.0)))
            df[f'd_dist_{name}'] = d_dist
            if name in ('jfk', 'lga', 'ewr'):
                df[f'p_near_{name}'] = (p_dist < 3.0).astype('float64')
                df[f'd_near_{name}'] = (d_dist < 3.0).astype('float64')
                df[f'trip_near_{name}'] = ((p_dist < 3.0) | (d_dist < 3.0)).astype('float64')
        p_near_midtown = (df['p_dist_midtown'] < 3.0).astype('float64')
        d_near_midtown = (df['d_dist_midtown'] < 3.0).astype('float64')
        df['is_jfk_midtown'] = ((df['p_near_jfk'] == 1.0) & (d_near_midtown == 1.0) | (p_near_midtown == 1.0) & (df['d_near_jfk'] == 1.0)).astype('float64')
        df['is_ewr_midtown'] = ((df['p_near_ewr'] == 1.0) & (d_near_midtown == 1.0) | (p_near_midtown == 1.0) & (df['d_near_ewr'] == 1.0)).astype('float64')
        df['passenger_count'] = X['passenger_count'].astype('float64')
        dt = pd.to_datetime(X['pickup_datetime'].astype(str).str.slice(0, 19), format='%Y-%m-%d %H:%M:%S', errors='coerce')
        hour = dt.dt.hour.fillna(12).astype('float64')
        dow = dt.dt.dayofweek.fillna(2).astype('float64')
        month = dt.dt.month.fillna(6).astype('float64')
        year = dt.dt.year.fillna(2012).astype('float64')
        day = dt.dt.day.fillna(15).astype('float64')
        df['year'] = year
        df['frac_year'] = year + (month - 1.0) / 12.0 + (day - 1.0) / 365.25
        hike_date = pd.Timestamp('2012-09-04')
        is_post_hike_2012 = (dt >= hike_date).astype('float64')
        df['is_post_hike_2012'] = is_post_hike_2012
        df['haversine_post_hike'] = haversine_km * is_post_hike_2012
        df['rot_manhattan_post_hike'] = rot_manhattan * is_post_hike_2012
        df['is_rush_hour'] = ((dow < 5.0) & (hour >= 16.0) & (hour < 20.0)).astype('float64')
        df['is_overnight'] = ((hour >= 20.0) | (hour < 6.0)).astype('float64')
        df['sin_hour'] = np.sin(2.0 * np.pi * hour / 24.0)
        df['cos_hour'] = np.cos(2.0 * np.pi * hour / 24.0)
        df['sin_dow'] = np.sin(2.0 * np.pi * dow / 7.0)
        df['cos_dow'] = np.cos(2.0 * np.pi * dow / 7.0)
        df['sin_month'] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        df['cos_month'] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        df['is_weekend'] = (dow >= 5.0).astype('float64')
        norm_p_lon = (p_lon - -73.98) * 10.0
        norm_p_lat = (p_lat - 40.75) * 10.0
        norm_d_lon = (d_lon - -73.98) * 10.0
        norm_d_lat = (d_lat - 40.75) * 10.0
        norm_p_rot_x = cos_ang * norm_p_lon - sin_ang * norm_p_lat
        norm_p_rot_y = sin_ang * norm_p_lon + cos_ang * norm_p_lat
        norm_d_rot_x = cos_ang * norm_d_lon - sin_ang * norm_d_lat
        norm_d_rot_y = sin_ang * norm_d_lon + cos_ang * norm_d_lat
        fourier_targets = {'p_lon': norm_p_lon, 'p_lat': norm_p_lat, 'd_lon': norm_d_lon, 'd_lat': norm_d_lat, 'p_rot_x': norm_p_rot_x, 'p_rot_y': norm_p_rot_y, 'd_rot_x': norm_d_rot_x, 'd_rot_y': norm_d_rot_y}
        for k in range(int(self.n_fourier_octaves)):
            freq = 2.0 ** k * np.pi
            for var_name, series_val in fourier_targets.items():
                df[f'fourier_sin_k{k}_{var_name}'] = np.sin(freq * series_val)
                df[f'fourier_cos_k{k}_{var_name}'] = np.cos(freq * series_val)
        return df.fillna(0.0)

    def fit(self, X, y=None):
        raw = self._extract_raw(X)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(raw)
        self.columns_ = list(raw.columns)
        return self

    def transform(self, X):
        raw = self._extract_raw(X)
        scaled = self.scaler_.transform(raw)
        return pd.DataFrame(scaled, columns=self.columns_, index=X.index)

class ResNetBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        return x + self.block(x)

class TabularResNetModule(nn.Module):

    def __init__(self, in_features, hidden_dim=256, n_blocks=3, dropout=0.05, bypass_dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        if bypass_dropout > 0.0:
            self.linear_bypass = nn.Sequential(nn.Dropout(bypass_dropout), nn.Linear(in_features, 1))
        else:
            self.linear_bypass = nn.Linear(in_features, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        deep_out = self.head(h)
        wide_out = self.linear_bypass(x)
        return deep_out + wide_out

class TorchResNetRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, hidden_dim=256, n_blocks=3, lr=0.001, batch_size=2048, epochs=15, dropout=0.05, bypass_dropout=0.1, random_state=42):
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.bypass_dropout = bypass_dropout
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ = device
        X_vals = X.values.astype(np.float32) if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        y_vals = y.values.astype(np.float32) if hasattr(y, 'values') else np.asarray(y, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X_vals), torch.from_numpy(y_vals))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.net_ = TabularResNetModule(in_features=X_vals.shape[1], hidden_dim=self.hidden_dim, n_blocks=self.n_blocks, dropout=self.dropout, bypass_dropout=self.bypass_dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        self.net_.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                bx = bx.to(device)
                by = by.to(device)
                optimizer.zero_grad()
                out = self.net_(bx).squeeze(-1)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, X):
        self.net_.eval()
        X_vals = X.values.astype(np.float32) if hasattr(X, 'values') else np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.from_numpy(X_vals))
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)
        preds = []
        with torch.no_grad():
            for bx, in loader:
                bx = bx.to(self.device_)
                out = self.net_(bx).squeeze(-1)
                preds.append(out.cpu().numpy())
        pred_arr = np.concatenate(preds, axis=0)
        return np.clip(pred_arr, 2.5, 500.0)

class TriEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        w_lgbm=0.45,
        w_xgb=0.35,
        w_resnet=0.20,
        n_estimators_lgbm=500,
        num_leaves_lgbm=63,
        lr_lgbm=0.05,
        n_estimators_xgb=400,
        max_depth_xgb=7,
        lr_xgb=0.05,
        resnet_hidden_dim=256,
        resnet_blocks=3,
        resnet_lr=0.001,
        resnet_batch_size=2048,
        resnet_epochs=15,
        resnet_dropout=0.05,
        resnet_bypass_dropout=0.1,
        random_state=42,
    ):
        self.w_lgbm = w_lgbm
        self.w_xgb = w_xgb
        self.w_resnet = w_resnet
        self.n_estimators_lgbm = n_estimators_lgbm
        self.num_leaves_lgbm = num_leaves_lgbm
        self.lr_lgbm = lr_lgbm
        self.n_estimators_xgb = n_estimators_xgb
        self.max_depth_xgb = max_depth_xgb
        self.lr_xgb = lr_xgb
        self.resnet_hidden_dim = resnet_hidden_dim
        self.resnet_blocks = resnet_blocks
        self.resnet_lr = resnet_lr
        self.resnet_batch_size = resnet_batch_size
        self.resnet_epochs = resnet_epochs
        self.resnet_dropout = resnet_dropout
        self.resnet_bypass_dropout = resnet_bypass_dropout
        self.random_state = random_state

    def fit(self, X, y):
        self.lgbm_ = LGBMRegressor(
            n_estimators=self.n_estimators_lgbm,
            learning_rate=self.lr_lgbm,
            num_leaves=self.num_leaves_lgbm,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=8,
        )
        self.lgbm_.fit(X, y)

        self.xgb_ = XGBRegressor(
            n_estimators=self.n_estimators_xgb,
            learning_rate=self.lr_xgb,
            max_depth=self.max_depth_xgb,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=self.random_state,
            n_jobs=8,
        )
        self.xgb_.fit(X, y)

        self.resnet_ = TorchResNetRegressor(
            hidden_dim=self.resnet_hidden_dim,
            n_blocks=self.resnet_blocks,
            lr=self.resnet_lr,
            batch_size=self.resnet_batch_size,
            epochs=self.resnet_epochs,
            dropout=self.resnet_dropout,
            bypass_dropout=self.resnet_bypass_dropout,
            random_state=self.random_state,
        )
        self.resnet_.fit(X, y)
        return self

    def predict(self, X):
        p_lgbm = self.lgbm_.predict(X)
        p_xgb = self.xgb_.predict(X)
        p_resnet = self.resnet_.predict(X)
        blended = self.w_lgbm * p_lgbm + self.w_xgb * p_xgb + self.w_resnet * p_resnet
        return np.clip(blended, 2.5, 500.0)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']
    preprocessor = TabularPreprocessor(n_fourier_octaves=6)
    geo_features = X.skb.apply(preprocessor).skb.set_name('geo_time_features')
    models = {
        'balanced_trees': TriEnsembleRegressor(w_lgbm=0.45, w_xgb=0.35, w_resnet=0.20, random_state=42),
        'equal_trees': TriEnsembleRegressor(w_lgbm=0.40, w_xgb=0.40, w_resnet=0.20, random_state=42),
    }
    chosen_model = skrub.choose_from(models, name='weights')
    pred = geo_features.skb.apply(chosen_model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': scoring}