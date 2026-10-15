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
            nn.BatchNorm1d(hidden_dim),
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
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

class TriEnsembleRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, lgb_weight=0.5, xgb_weight=0.35, resnet_weight=0.15, hidden_dim=256, n_blocks=3, lr=0.001, epochs=15, batch_size=2048, dropout=0.05, random_state=42):
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight
        self.resnet_weight = resnet_weight
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

    def fit(self, X, y):
        self.lgb_model_ = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.08,
            num_leaves=63,
            random_state=self.random_state,
            n_jobs=8,
        )
        self.lgb_model_.fit(X, y)

        xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.xgb_model_ = XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device=xgb_device,
            random_state=self.random_state,
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
            dropout=self.dropout,
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
        return self.lgb_weight * p_lgb + self.xgb_weight * p_xgb + self.resnet_weight * p_resnet

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

    dx_km = d_lon_signed * 84.3
    dy_km = d_lat_signed * 111.0
    rot_manhattan_km = (dx_km * 0.8746197 + dy_km * 0.4848096).abs() + (-dx_km * 0.4848096 + dy_km * 0.8746197).abs()
    bearing = d_lat_signed.skb.apply_func(np.arctan2, d_lon_signed)

    p_jfk = (((p_lat - 40.6413) * 111.0) ** 2 + ((p_lon - -73.7781) * 84.3) ** 2) ** 0.5
    d_jfk = (((d_lat - 40.6413) * 111.0) ** 2 + ((d_lon - -73.7781) * 84.3) ** 2) ** 0.5
    p_lga = (((p_lat - 40.7769) * 111.0) ** 2 + ((p_lon - -73.874) * 84.3) ** 2) ** 0.5
    d_lga = (((d_lat - 40.7769) * 111.0) ** 2 + ((d_lon - -73.874) * 84.3) ** 2) ** 0.5
    p_ewr = (((p_lat - 40.6895) * 111.0) ** 2 + ((p_lon - -74.1745) * 84.3) ** 2) ** 0.5
    d_ewr = (((d_lat - 40.6895) * 111.0) ** 2 + ((d_lon - -74.1745) * 84.3) ** 2) ** 0.5
    p_midtown = (((p_lat - 40.758) * 111.0) ** 2 + ((p_lon - -73.9855) * 84.3) ** 2) ** 0.5
    d_midtown = (((d_lat - 40.758) * 111.0) ** 2 + ((d_lon - -73.9855) * 84.3) ** 2) ** 0.5

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

    p_jfk_close = (p_jfk < 3.0).astype('float64')
    d_jfk_close = (d_jfk < 3.0).astype('float64')
    jfk_trip = ((p_jfk < 3.0).fillna(False) | (d_jfk < 3.0).fillna(False)).astype('float64')
    lga_trip = ((p_lga < 2.5).fillna(False) | (d_lga < 2.5).fillna(False)).astype('float64')
    ewr_trip = ((p_ewr < 3.0).fillna(False) | (d_ewr < 3.0).fillna(False)).astype('float64')

    mid_lat = (p_lat + d_lat) / 2.0
    mid_lon = (p_lon + d_lon) / 2.0

    p_manhattan = ((p_lat >= 40.7).fillna(False) & (p_lat <= 40.83).fillna(False) & (p_lon >= -74.02).fillna(False) & (p_lon <= -73.93).fillna(False)).astype('float64')
    d_manhattan = ((d_lat >= 40.7).fillna(False) & (d_lat <= 40.83).fillna(False) & (d_lon >= -74.02).fillna(False) & (d_lon <= -73.93).fillna(False)).astype('float64')
    intra_manhattan = ((p_manhattan > 0.5).fillna(False) & (d_manhattan > 0.5).fillna(False)).astype('float64')
    jfk_flat_rate = ((p_jfk < 3.0).fillna(False) & (d_manhattan > 0.5).fillna(False) | (d_jfk < 3.0).fillna(False) & (p_manhattan > 0.5).fillna(False)).astype('float64')

    decimal_year = year.astype('float64') + (month.astype('float64') - 1.0) / 12.0 + (day.astype('float64') - 1.0) / 365.25

    hour_angle = hour.astype('float64') * (2.0 * np.pi / 24.0)
    sin_hour = hour_angle.skb.apply_func(np.sin)
    cos_hour = hour_angle.skb.apply_func(np.cos)

    statutory_base_charge = 2.50 + 0.50 * is_post_2009 + 0.50 * overnight_surcharge + 1.00 * rush_hour_surcharge
    statutory_meter_rate = 1.243 + 0.310 * is_post_2012
    statutory_meter_fare = statutory_base_charge + statutory_meter_rate * rot_manhattan_km
    jfk_statutory_fare = jfk_flat_rate * (45.0 + 7.0 * is_post_2012)
    ewr_surcharge = ewr_trip * 15.0

    base_features = X[['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].assign(
        d_lat_abs=d_lat_abs,
        d_lon_abs=d_lon_abs,
        d_lat=d_lat_signed,
        d_lon=d_lon_signed,
        lat_km=lat_km,
        lon_km=lon_km,
        euclidean_km=euclidean_km,
        manhattan_km=manhattan_km,
        rot_manhattan_km=rot_manhattan_km,
        bearing=bearing,
        p_jfk=p_jfk,
        d_jfk=d_jfk,
        p_lga=p_lga,
        d_lga=d_lga,
        p_ewr=p_ewr,
        d_ewr=d_ewr,
        p_midtown=p_midtown,
        d_midtown=d_midtown,
        year=year,
        month=month,
        day=day,
        hour=hour,
        weekday=weekday,
        is_post_2012=is_post_2012,
        overnight_surcharge=overnight_surcharge,
        rush_hour_surcharge=rush_hour_surcharge,
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
    )

    cross_borough_trip = (p_manhattan != d_manhattan).astype('float64')
    min_p_airport = p_jfk.skb.apply_func(np.minimum, p_lga).skb.apply_func(np.minimum, p_ewr)
    min_d_airport = d_jfk.skb.apply_func(np.minimum, d_lga).skb.apply_func(np.minimum, d_ewr)

    extended_features = base_features.assign(
        cross_borough_trip=cross_borough_trip,
        min_p_airport=min_p_airport,
        min_d_airport=min_d_airport,
    )

    feature_choice = skrub.choose_from(
        {
            'statutory_tariff_features': base_features,
            'statutory_and_borough_crossing_features': extended_features,
        },
        name='feature_variant',
    )
    features = skrub.as_data_op(feature_choice).skb.set_name('features')

    model = TriEnsembleRegressor(lgb_weight=0.5, xgb_weight=0.35, resnet_weight=0.15)
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': scoring}