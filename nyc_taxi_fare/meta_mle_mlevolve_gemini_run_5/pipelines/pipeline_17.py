import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
import skrub

class ResNetBlock(nn.Module):

    def __init__(self, hidden_dim, dropout=0.05):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class TabularResNet(nn.Module):

    def __init__(self, in_features, hidden_dim=256, n_blocks=3, dropout=0.05):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

class TriEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        lgb_weight=0.45,
        xgb_weight=0.45,
        resnet_weight=0.1,
        n_estimators=1000,
        learning_rate=0.030,
        num_leaves=192,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        subsample_freq=1,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        hidden_dim=256,
        n_blocks=3,
        lr=0.001,
        epochs=15,
        batch_size=2048,
        dropout=0.05,
        random_state=42
    ):
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        self.resnet_weight = resnet_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.subsample_freq = subsample_freq
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

    def fit(self, X, y):
        self.lgb_model_ = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            subsample_freq=self.subsample_freq,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=8
        )
        self.lgb_model_.fit(X, y)

        xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xgb_model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            tree_method='hist',
            device=xgb_device,
            random_state=self.random_state
        )
        self.xgb_model_.fit(X, y)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.scaler_ = StandardScaler()
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.to_numpy(dtype=np.float32).ravel()
        else:
            y_arr = np.asarray(y, dtype=np.float32).ravel()

        X_scaled = self.scaler_.fit_transform(X_arr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_features = X_scaled.shape[1]
        self.resnet_model_ = TabularResNet(
            in_features=in_features,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout
        ).to(device)

        dataset = TensorDataset(torch.from_numpy(X_scaled), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.resnet_model_.parameters(), lr=self.lr, weight_decay=0.0001)
        criterion = nn.MSELoss()

        self.resnet_model_.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = self.resnet_model_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        p_lgb = self.lgb_model_.predict(X)
        p_xgb = self.xgb_model_.predict(X)

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X_arr)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet_model_.eval()
        preds_resnet = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), self.batch_size):
                batch_x = torch.from_numpy(X_scaled[i:i + self.batch_size]).to(device)
                p = self.resnet_model_(batch_x)
                preds_resnet.append(p.cpu().numpy())
        p_resnet = np.concatenate(preds_resnet, axis=0)

        p_ensemble = self.lgb_weight * p_lgb + self.xgb_weight * p_xgb + self.resnet_weight * p_resnet
        return np.clip(p_ensemble, 2.5, None)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    p_lat = X['pickup_latitude']
    p_lon = X['pickup_longitude']
    d_lat = X['dropoff_latitude']
    d_lon = X['dropoff_longitude']

    d_lat_abs = (d_lat - p_lat).abs()
    d_lon_abs = (d_lon - p_lon).abs()
    d_lat_signed = d_lat - p_lat
    d_lon_signed = d_lon - p_lon

    lat_km = d_lat_abs * 111.0
    lon_km = d_lon_abs * 84.3
    euclidean_km = (lat_km ** 2 + lon_km ** 2) ** 0.5
    manhattan_km = lat_km + lon_km

    cos_grid = 0.8746197
    sin_grid = 0.4848096

    dx_km = d_lon_signed * 84.3
    dy_km = d_lat_signed * 111.0

    crosstown_signed_km = dx_km * cos_grid + dy_km * sin_grid
    avenue_signed_km = -dx_km * sin_grid + dy_km * cos_grid
    crosstown_km = crosstown_signed_km.abs()
    avenue_km = avenue_signed_km.abs()
    rot_manhattan_km = crosstown_km + avenue_km

    p_x_km = (p_lon - -73.9855) * 84.3
    p_y_km = (p_lat - 40.758) * 111.0
    rot_p_x = p_x_km * cos_grid + p_y_km * sin_grid
    rot_p_y = -p_x_km * sin_grid + p_y_km * cos_grid

    d_x_km = (d_lon - -73.9855) * 84.3
    d_y_km = (d_lat - 40.758) * 111.0
    rot_d_x = d_x_km * cos_grid + d_y_km * sin_grid
    rot_d_y = -d_x_km * sin_grid + d_y_km * cos_grid

    rot_mid_x = (rot_p_x + rot_d_x) / 2.0
    rot_mid_y = (rot_p_y + rot_d_y) / 2.0

    bearing = d_lat_signed.skb.apply_func(np.arctan2, d_lon_signed)
    sin_bearing = bearing.skb.apply_func(np.sin)
    cos_bearing = bearing.skb.apply_func(np.cos)

    p_jfk = (((p_lat - 40.6413) * 111.0) ** 2 + ((p_lon - -73.7781) * 84.3) ** 2) ** 0.5
    d_jfk = (((d_lat - 40.6413) * 111.0) ** 2 + ((d_lon - -73.7781) * 84.3) ** 2) ** 0.5
    p_lga = (((p_lat - 40.7769) * 111.0) ** 2 + ((p_lon - -73.874) * 84.3) ** 2) ** 0.5
    d_lga = (((d_lat - 40.7769) * 111.0) ** 2 + ((d_lon - -73.874) * 84.3) ** 2) ** 0.5
    p_ewr = (((p_lat - 40.6895) * 111.0) ** 2 + ((p_lon - -74.1745) * 84.3) ** 2) ** 0.5
    d_ewr = (((d_lat - 40.6895) * 111.0) ** 2 + ((d_lon - -74.1745) * 84.3) ** 2) ** 0.5

    p_midtown = (((p_lat - 40.758) * 111.0) ** 2 + ((p_lon - -73.9855) * 84.3) ** 2) ** 0.5
    d_midtown = (((d_lat - 40.758) * 111.0) ** 2 + ((d_lon - -73.9855) * 84.3) ** 2) ** 0.5
    mid_lat = (p_lat + d_lat) / 2.0
    mid_lon = (p_lon + d_lon) / 2.0
    mid_midtown = (((mid_lat - 40.758) * 111.0) ** 2 + ((mid_lon - -73.9855) * 84.3) ** 2) ** 0.5

    p_fidi = (((p_lat - 40.7075) * 111.0) ** 2 + ((p_lon - -74.009) * 84.3) ** 2) ** 0.5
    d_fidi = (((d_lat - 40.7075) * 111.0) ** 2 + ((d_lon - -74.009) * 84.3) ** 2) ** 0.5

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    hour = dt.dt.hour
    weekday = dt.dt.dayofweek

    is_post_2012 = ((year > 2012).fillna(False) | (year == 2012).fillna(False) & (month >= 9).fillna(False)).astype('float64')
    is_post_2009 = ((year > 2009).fillna(False) | (year == 2009).fillna(False) & (month >= 11).fillna(False)).astype('float64')
    overnight_surcharge = ((hour >= 20).fillna(False) | (hour < 6).fillna(False)).astype('float64')
    rush_hour_surcharge = ((weekday < 5).fillna(False) & (hour >= 16).fillna(False) & (hour < 20).fillna(False)).astype('float64')
    is_morning_peak = ((weekday < 5).fillna(False) & (hour >= 7).fillna(False) & (hour <= 9).fillna(False)).astype('float64')

    p_jfk_close = (p_jfk < 3.0).astype('float64')
    d_jfk_close = (d_jfk < 3.0).astype('float64')
    jfk_trip = ((p_jfk < 3.0).fillna(False) | (d_jfk < 3.0).fillna(False)).astype('float64')
    lga_trip = ((p_lga < 2.5).fillna(False) | (d_lga < 2.5).fillna(False)).astype('float64')
    ewr_trip = ((p_ewr < 3.0).fillna(False) | (d_ewr < 3.0).fillna(False)).astype('float64')

    p_manhattan = ((p_lat >= 40.7).fillna(False) & (p_lat <= 40.83).fillna(False) & (p_lon >= -74.02).fillna(False) & (p_lon <= -73.93).fillna(False)).astype('float64')
    d_manhattan = ((d_lat >= 40.7).fillna(False) & (d_lat <= 40.83).fillna(False) & (d_lon >= -74.02).fillna(False) & (d_lon <= -73.93).fillna(False)).astype('float64')
    intra_manhattan = ((p_manhattan > 0.5).fillna(False) & (d_manhattan > 0.5).fillna(False)).astype('float64')
    manhattan_rush = intra_manhattan * rush_hour_surcharge
    manhattan_morning_peak = intra_manhattan * is_morning_peak

    jfk_flat_rate = ((p_jfk < 3.0).fillna(False) & (d_manhattan > 0.5).fillna(False) | (d_jfk < 3.0).fillna(False) & (p_manhattan > 0.5).fillna(False)).astype('float64')
    statutory_base_charge = 2.5 + 0.5 * is_post_2009 + 0.5 * overnight_surcharge + 1.0 * rush_hour_surcharge
    statutory_meter_rate = 1.243 + 0.31 * is_post_2012
    statutory_meter_fare = statutory_base_charge + statutory_meter_rate * rot_manhattan_km
    jfk_statutory_fare = jfk_flat_rate * (45.0 + 7.0 * is_post_2012)
    ewr_surcharge = ewr_trip * (15.0 + 2.5 * is_post_2012)
    composite_statutory_fare = (
        jfk_flat_rate * (45.0 + 7.0 * is_post_2012 + 0.5 * is_post_2009 + 0.5 * overnight_surcharge + 1.0 * rush_hour_surcharge)
        + (1.0 - jfk_flat_rate) * (statutory_meter_fare + ewr_surcharge)
    )

    crosstown_ratio = crosstown_km / (rot_manhattan_km + 0.05)
    is_zero_dist = (euclidean_km < 0.05).astype('float64')

    decimal_year = year.astype('float64') + (month.astype('float64') - 1.0) / 12.0 + (day.astype('float64') - 1.0) / 365.25
    hour_angle = hour.astype('float64') * (2.0 * np.pi / 24.0)
    sin_hour = hour_angle.skb.apply_func(np.sin)
    cos_hour = hour_angle.skb.apply_func(np.cos)
    is_weekend = (weekday >= 5).astype('float64')
    month_angle = (month.astype('float64') - 1.0) * (2.0 * np.pi / 12.0)
    sin_month = month_angle.skb.apply_func(np.sin)
    cos_month = month_angle.skb.apply_func(np.cos)

    feat_a = X[['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].assign(
        d_lat_abs=d_lat_abs,
        d_lon_abs=d_lon_abs,
        d_lat=d_lat_signed,
        d_lon=d_lon_signed,
        lat_km=lat_km,
        lon_km=lon_km,
        euclidean_km=euclidean_km,
        manhattan_km=manhattan_km,
        crosstown_signed_km=crosstown_signed_km,
        avenue_signed_km=avenue_signed_km,
        crosstown_km=crosstown_km,
        avenue_km=avenue_km,
        rot_manhattan_km=rot_manhattan_km,
        rot_p_x=rot_p_x,
        rot_p_y=rot_p_y,
        rot_d_x=rot_d_x,
        rot_d_y=rot_d_y,
        rot_mid_x=rot_mid_x,
        rot_mid_y=rot_mid_y,
        bearing=bearing,
        sin_bearing=sin_bearing,
        cos_bearing=cos_bearing,
        p_jfk=p_jfk,
        d_jfk=d_jfk,
        p_lga=p_lga,
        d_lga=d_lga,
        p_ewr=p_ewr,
        d_ewr=d_ewr,
        p_midtown=p_midtown,
        d_midtown=d_midtown,
        mid_midtown=mid_midtown,
        p_fidi=p_fidi,
        d_fidi=d_fidi,
        year=year,
        month=month,
        day=day,
        hour=hour,
        weekday=weekday,
        is_weekend=is_weekend,
        sin_month=sin_month,
        cos_month=cos_month,
        is_post_2012=is_post_2012,
        overnight_surcharge=overnight_surcharge,
        rush_hour_surcharge=rush_hour_surcharge,
        is_morning_peak=is_morning_peak,
        manhattan_rush=manhattan_rush,
        manhattan_morning_peak=manhattan_morning_peak,
        p_jfk_close=p_jfk_close,
        d_jfk_close=d_jfk_close,
        jfk_trip=jfk_trip,
        lga_trip=lga_trip,
        ewr_trip=ewr_trip,
        mid_lat=mid_lat,
        mid_lon=mid_lon,
        p_manhattan=p_manhattan,
        d_manhattan=d_manhattan,
        intra_manhattan=intra_manhattan,
        jfk_flat_rate=jfk_flat_rate,
        decimal_year=decimal_year,
        sin_hour=sin_hour,
        cos_hour=cos_hour,
        statutory_base_charge=statutory_base_charge,
        statutory_meter_rate=statutory_meter_rate,
        statutory_meter_fare=statutory_meter_fare,
        jfk_statutory_fare=jfk_statutory_fare,
        ewr_surcharge=ewr_surcharge,
        composite_statutory_fare=composite_statutory_fare,
        crosstown_ratio=crosstown_ratio,
        is_zero_dist=is_zero_dist
    )

    detour_ratio = rot_manhattan_km / (euclidean_km + 0.05)
    p_queens = ((p_lat >= 40.68).fillna(False) & (p_lat <= 40.8).fillna(False) & (p_lon >= -73.93).fillna(False) & (p_lon <= -73.7).fillna(False)).astype('float64')
    d_queens = ((d_lat >= 40.68).fillna(False) & (d_lat <= 40.8).fillna(False) & (d_lon >= -73.93).fillna(False) & (d_lon <= -73.7).fillna(False)).astype('float64')
    manhattan_to_queens = ((p_manhattan > 0.5).fillna(False) & (d_queens > 0.5).fillna(False) | (p_queens > 0.5).fillna(False) & (d_manhattan > 0.5).fillna(False)).astype('float64')

    p_brooklyn = ((p_lat >= 40.57).fillna(False) & (p_lat <= 40.7).fillna(False) & (p_lon >= -74.04).fillna(False) & (p_lon <= -73.85).fillna(False)).astype('float64')
    d_brooklyn = ((d_lat >= 40.57).fillna(False) & (d_lat <= 40.7).fillna(False) & (d_lon >= -74.04).fillna(False) & (d_lon <= -73.85).fillna(False)).astype('float64')
    manhattan_to_brooklyn = ((p_manhattan > 0.5).fillna(False) & (d_brooklyn > 0.5).fillna(False) | (p_brooklyn > 0.5).fillna(False) & (d_manhattan > 0.5).fillna(False)).astype('float64')

    feat_b = feat_a.assign(
        detour_ratio=detour_ratio,
        manhattan_to_queens=manhattan_to_queens,
        manhattan_to_brooklyn=manhattan_to_brooklyn
    )

    features = feat_b.skb.set_name('features')

    model_choice = skrub.choose_from({
        'scaled_capacity_leaves_192': TriEnsembleRegressor(
            lgb_weight=0.45,
            xgb_weight=0.45,
            resnet_weight=0.10,
            n_estimators=1000,
            learning_rate=0.030,
            num_leaves=192,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            subsample_freq=1,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        ),
        'deep_capacity_leaves_224': TriEnsembleRegressor(
            lgb_weight=0.46,
            xgb_weight=0.46,
            resnet_weight=0.08,
            n_estimators=1150,
            learning_rate=0.025,
            num_leaves=224,
            max_depth=10,
            subsample=0.8,
            colsample_bytree=0.8,
            subsample_freq=1,
            min_child_samples=50,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42
        )
    }, name='model_variant')

    pred = features.skb.apply(model_choice, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': scoring}