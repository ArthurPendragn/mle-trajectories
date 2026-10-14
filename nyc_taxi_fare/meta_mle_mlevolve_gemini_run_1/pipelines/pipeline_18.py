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

    def __init__(self, n_fourier_octaves=6, include_log_dist=True):
        self.n_fourier_octaves = n_fourier_octaves
        self.include_log_dist = include_log_dist

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
        df['rot_delta_x'] = d_rot_x - p_rot_x
        df['rot_delta_y'] = d_rot_y - p_rot_y
        df['mid_rot_x'] = 0.5 * (p_rot_x + d_rot_x)
        df['mid_rot_y'] = 0.5 * (p_rot_y + d_rot_y)

        rot_manhattan = rot_dist_x + rot_dist_y
        df['rot_manhattan'] = rot_manhattan
        df['crosstown_ratio'] = rot_dist_x / (rot_manhattan + 1e-4)

        euclidean = np.sqrt(df['d_lat_abs'] ** 2 + df['d_lon_abs'] ** 2)
        df['euclidean'] = euclidean

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
        bearing = np.arctan2(y_bear, x_bear)
        df['bearing'] = bearing
        df['sin_bearing'] = np.sin(bearing)
        df['cos_bearing'] = np.cos(bearing)

        landmarks = {
            'jfk': (-73.7781, 40.6413),
            'lga': (-73.8740, 40.7769),
            'ewr': (-74.1745, 40.6895),
            'midtown': (-73.9851, 40.7580),
            'downtown': (-74.0090, 40.7128),
            'brooklyn': (-73.9857, 40.6928),
            'penn': (-73.9935, 40.7505),
            'grand_central': (-73.9772, 40.7527),
        }
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
            df[f'trip_min_dist_{name}'] = np.minimum(p_dist, d_dist)

            if name in ('jfk', 'lga', 'ewr'):
                df[f'p_near_{name}'] = (p_dist < 3.0).astype('float64')
                df[f'd_near_{name}'] = (d_dist < 3.0).astype('float64')
                df[f'trip_near_{name}'] = ((p_dist < 3.0) | (d_dist < 3.0)).astype('float64')

        p_near_midtown = (df['p_dist_midtown'] < 3.0).astype('float64')
        d_near_midtown = (df['d_dist_midtown'] < 3.0).astype('float64')
        df['is_jfk_midtown'] = (
            ((df['p_near_jfk'] == 1.0) & (d_near_midtown == 1.0))
            | ((p_near_midtown == 1.0) & (df['d_near_jfk'] == 1.0))
        ).astype('float64')
        df['is_ewr_midtown'] = (
            ((df['p_near_ewr'] == 1.0) & (d_near_midtown == 1.0))
            | ((p_near_midtown == 1.0) & (df['d_near_ewr'] == 1.0))
        ).astype('float64')

        p_in_manhattan = ((p_lon >= -74.03) & (p_lon <= -73.93) & (p_lat >= 40.70) & (p_lat <= 40.88)).astype('float64')
        d_in_manhattan = ((d_lon >= -74.03) & (d_lon <= -73.93) & (d_lat >= 40.70) & (d_lat <= 40.88)).astype('float64')
        df['p_in_manhattan'] = p_in_manhattan
        df['d_in_manhattan'] = d_in_manhattan
        df['is_intra_manhattan'] = ((p_in_manhattan == 1.0) & (d_in_manhattan == 1.0)).astype('float64')

        is_jfk_manhattan = (
            ((df['p_near_jfk'] == 1.0) & (d_in_manhattan == 1.0))
            | ((p_in_manhattan == 1.0) & (df['d_near_jfk'] == 1.0))
        ).astype('float64')
        df['is_jfk_manhattan'] = is_jfk_manhattan
        df['is_cross_river'] = (p_in_manhattan != d_in_manhattan).astype('float64')
        df['is_zero_dist'] = (haversine_km < 0.05).astype('float64')
        df['is_ewr_trip'] = df['trip_near_ewr']
        df['is_lga_manhattan'] = (
            ((df['p_near_lga'] == 1.0) & (d_in_manhattan == 1.0))
            | ((p_in_manhattan == 1.0) & (df['d_near_lga'] == 1.0))
        ).astype('float64')

        p_east_cp = (p_lon > -73.965) & (p_lat >= 40.764) & (p_lat <= 40.800)
        d_west_cp = (d_lon < -73.975) & (d_lat >= 40.764) & (d_lat <= 40.800)
        d_east_cp = (d_lon > -73.965) & (d_lat >= 40.764) & (d_lat <= 40.800)
        p_west_cp = (p_lon < -73.975) & (p_lat >= 40.764) & (p_lat <= 40.800)
        df['is_cross_central_park'] = ((p_east_cp & d_west_cp) | (p_west_cp & d_east_cp)).astype('float64')

        df['passenger_count'] = X['passenger_count'].astype('float64')

        dt = pd.to_datetime(X['pickup_datetime'].astype(str).str.slice(0, 19), format='%Y-%m-%d %H:%M:%S', errors='coerce')
        hour = dt.dt.hour.fillna(12).astype('float64')
        dow = dt.dt.dayofweek.fillna(2).astype('float64')
        month = dt.dt.month.fillna(6).astype('float64')
        year = dt.dt.year.fillna(2012).astype('float64')
        day = dt.dt.day.fillna(15).astype('float64')

        df['hour'] = hour
        df['dow'] = dow
        df['month'] = month
        df['year'] = year
        df['frac_year'] = year + (month - 1.0) / 12.0 + (day - 1.0) / 365.25

        hike_date = pd.Timestamp('2012-09-04')
        is_post_hike_2012 = (dt >= hike_date).astype('float64')
        df['is_post_hike_2012'] = is_post_hike_2012
        df['haversine_post_hike'] = haversine_km * is_post_hike_2012
        df['rot_manhattan_post_hike'] = rot_manhattan * is_post_hike_2012
        df['rot_dist_x_post_hike'] = rot_dist_x * is_post_hike_2012
        df['rot_dist_y_post_hike'] = rot_dist_y * is_post_hike_2012
        df['jfk_manhattan_post_hike'] = is_jfk_manhattan * is_post_hike_2012
        df['lga_manhattan_post_hike'] = df['is_lga_manhattan'] * is_post_hike_2012
        df['ewr_post_hike'] = df['is_ewr_trip'] * is_post_hike_2012

        mta_date = pd.Timestamp('2009-11-01')
        df['is_post_mta'] = (dt >= mta_date).astype('float64')

        t2015_date = pd.Timestamp('2015-01-01')
        df['is_post_2015'] = (dt >= t2015_date).astype('float64')

        is_rush_hour = ((dow < 5.0) & (hour >= 16.0) & (hour < 20.0)).astype('float64')
        df['is_rush_hour'] = is_rush_hour
        is_overnight = ((hour >= 20.0) | (hour < 6.0)).astype('float64')
        df['is_overnight'] = is_overnight
        df['haversine_rush'] = haversine_km * is_rush_hour
        df['rot_manhattan_rush'] = rot_manhattan * is_rush_hour
        df['rot_dist_x_rush'] = rot_dist_x * is_rush_hour
        df['haversine_overnight'] = haversine_km * is_overnight
        df['rot_manhattan_overnight'] = rot_manhattan * is_overnight

        is_morning_rush = ((dow < 5.0) & (hour >= 7.0) & (hour < 10.0)).astype('float64')
        df['is_morning_rush'] = is_morning_rush
        df['rot_manhattan_morning_rush'] = rot_manhattan * is_morning_rush
        df['is_jfk_post_hike_rush'] = is_jfk_manhattan * is_post_hike_2012 * is_rush_hour

        mile_rate = np.where(is_post_hike_2012 == 1.0, 1.5534, 1.2427)
        metered_surcharges = (
            0.50 * is_overnight +
            1.00 * is_rush_hour +
            0.50 * df['is_post_mta'] +
            0.30 * df['is_post_2015']
        )
        df['statutory_metered_fare'] = 2.50 + metered_surcharges + df['manhattan_km'] * mile_rate
        df['statutory_haversine_fare'] = 2.50 + metered_surcharges + df['haversine_km'] * mile_rate

        jfk_base = np.where(is_post_hike_2012 == 1.0, 52.0, 45.0)
        jfk_surcharges = (
            0.50 * df['is_post_mta'] +
            0.30 * df['is_post_2015'] +
            np.where(is_post_hike_2012 == 1.0, 4.50 * is_rush_hour, 0.0)
        )
        df['statutory_jfk_flat'] = jfk_base + jfk_surcharges
        df['statutory_composite_fare'] = np.where(is_jfk_manhattan == 1.0, df['statutory_jfk_flat'], df['statutory_metered_fare'])

        df['sin_hour'] = np.sin(2.0 * np.pi * hour / 24.0)
        df['cos_hour'] = np.cos(2.0 * np.pi * hour / 24.0)
        df['sin_dow'] = np.sin(2.0 * np.pi * dow / 7.0)
        df['cos_dow'] = np.cos(2.0 * np.pi * dow / 7.0)
        df['sin_month'] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        df['cos_month'] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        df['is_weekend'] = (dow >= 5.0).astype('float64')

        if self.include_log_dist:
            df['log_haversine'] = np.log1p(haversine_km)
            df['log_rot_manhattan'] = np.log1p(rot_manhattan)
        df['sqrt_haversine'] = np.sqrt(haversine_km)
        df['sqrt_rot_manhattan'] = np.sqrt(rot_manhattan)

        norm_p_lon = (p_lon - (-73.98)) * 10.0
        norm_p_lat = (p_lat - 40.75) * 10.0
        norm_d_lon = (d_lon - (-73.98)) * 10.0
        norm_d_lat = (d_lat - 40.75) * 10.0

        norm_p_rot_x = cos_ang * norm_p_lon - sin_ang * norm_p_lat
        norm_p_rot_y = sin_ang * norm_p_lon + cos_ang * norm_p_lat
        norm_d_rot_x = cos_ang * norm_d_lon - sin_ang * norm_d_lat
        norm_d_rot_y = sin_ang * norm_d_lon + cos_ang * norm_d_lat

        fourier_targets = {
            'p_lon': norm_p_lon,
            'p_lat': norm_p_lat,
            'd_lon': norm_d_lon,
            'd_lat': norm_d_lat,
            'p_rot_x': norm_p_rot_x,
            'p_rot_y': norm_p_rot_y,
            'd_rot_x': norm_d_rot_x,
            'd_rot_y': norm_d_rot_y,
        }
        for k in range(int(self.n_fourier_octaves)):
            freq = (2.0 ** k) * np.pi
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
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularResNetModule(nn.Module):

    def __init__(self, in_features, hidden_dim=256, n_blocks=3, dropout=0.05, bypass_dropout=0.0):
        super().__init__()
        self.input_layer = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        if bypass_dropout > 0.0:
            self.linear_bypass = nn.Sequential(
                nn.Dropout(bypass_dropout),
                nn.Linear(in_features, 1),
            )
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

    def __init__(
        self,
        hidden_dim=256,
        n_blocks=3,
        lr=0.001,
        batch_size=2048,
        epochs=20,
        dropout=0.05,
        bypass_dropout=0.0,
        random_state=42,
    ):
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

        self.net_ = TabularResNetModule(
            in_features=X_vals.shape[1],
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            bypass_dropout=self.bypass_dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
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
            for (bx,) in loader:
                bx = bx.to(self.device_)
                out = self.net_(bx).squeeze(-1)
                preds.append(out.cpu().numpy())

        pred_arr = np.concatenate(preds, axis=0)
        return np.clip(pred_arr, 2.5, 500.0)


class TriEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        w_lgbm=0.35,
        w_xgb=0.35,
        w_resnet=0.30,
        n_estimators_lgbm=800,
        num_leaves_lgbm=112,
        lr_lgbm=0.03,
        n_estimators_xgb=650,
        max_depth_xgb=8,
        lr_xgb=0.03,
        resnet_hidden_dim=256,
        resnet_blocks=3,
        resnet_lr=0.001,
        resnet_batch_size=2048,
        resnet_epochs=20,
        resnet_dropout=0.05,
        resnet_bypass_dropout=0.0,
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

    preprocessor = TabularPreprocessor(n_fourier_octaves=6, include_log_dist=True)
    geo_features = X.skb.apply(preprocessor).skb.set_name('geo_time_features')

    model_choices = {
        'balanced_trees_resnet': TriEnsembleRegressor(
            w_lgbm=0.35,
            w_xgb=0.35,
            w_resnet=0.30,
            n_estimators_lgbm=800,
            num_leaves_lgbm=112,
            lr_lgbm=0.03,
            n_estimators_xgb=650,
            max_depth_xgb=8,
            lr_xgb=0.03,
            resnet_epochs=20,
            resnet_bypass_dropout=0.0,
            random_state=42,
        ),
        'equal_tri_ensemble': TriEnsembleRegressor(
            w_lgbm=0.3333,
            w_xgb=0.3333,
            w_resnet=0.3334,
            n_estimators_lgbm=800,
            num_leaves_lgbm=112,
            lr_lgbm=0.03,
            n_estimators_xgb=650,
            max_depth_xgb=8,
            lr_xgb=0.03,
            resnet_epochs=20,
            resnet_bypass_dropout=0.0,
            random_state=42,
        ),
    }

    model = skrub.choose_from(model_choices, name='ensemble_weights')
    pred = geo_features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': scoring}