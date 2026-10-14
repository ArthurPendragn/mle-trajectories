import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class MLPNet(nn.Module):

    def __init__(self, in_features, hidden_dims=(256, 256, 128)):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchTabularRegressor(RegressorMixin, BaseEstimator):
    _estimator_type = 'regressor'

    def __init__(self, hidden_dims=(256, 256, 128), lr=0.002, weight_decay=1e-05, batch_size=4096, epochs=15, seed=42):
        self.hidden_dims = hidden_dims
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
        model = MLPNet(in_features=X_mat.shape[1], hidden_dims=self.hidden_dims).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.MSELoss()
        model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                bx, by = (bx.to(device), by.to(device))
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
    avenue_dist = dx_rot.abs().astype('float32')
    street_dist = dy_rot.abs().astype('float32')
    mid_longitude = ((X['pickup_longitude'] + X['dropoff_longitude']) * 0.5).astype('float32')
    mid_latitude = ((X['pickup_latitude'] + X['dropoff_latitude']) * 0.5).astype('float32')
    fractional_year = (year + (month - 1.0) / 12.0).astype('float32')
    bearing = dy_km.skb.apply_func(np.arctan2, dx_km)
    dist_pickup_jfk = (((X['pickup_latitude'] - 40.6413) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.7781) * 84.0) ** 2) ** 0.5
    dist_dropoff_jfk = (((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.7781) * 84.0) ** 2) ** 0.5
    dist_pickup_lga = (((X['pickup_latitude'] - 40.7769) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.874) * 84.0) ** 2) ** 0.5
    dist_dropoff_lga = (((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.874) * 84.0) ** 2) ** 0.5
    dist_pickup_ewr = (((X['pickup_latitude'] - 40.6895) * 111.0) ** 2 + ((X['pickup_longitude'] - -74.1745) * 84.0) ** 2) ** 0.5
    dist_dropoff_ewr = (((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2 + ((X['dropoff_longitude'] - -74.1745) * 84.0) ** 2) ** 0.5
    dist_pickup_midtown = (((X['pickup_latitude'] - 40.758) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.9855) * 84.0) ** 2) ** 0.5
    dist_dropoff_midtown = (((X['dropoff_latitude'] - 40.758) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.9855) * 84.0) ** 2) ** 0.5
    dist_pickup_downtown = (((X['pickup_latitude'] - 40.7075) * 111.0) ** 2 + ((X['pickup_longitude'] - -74.009) * 84.0) ** 2) ** 0.5
    dist_dropoff_downtown = (((X['dropoff_latitude'] - 40.7075) * 111.0) ** 2 + ((X['dropoff_longitude'] - -74.009) * 84.0) ** 2) ** 0.5
    dist_pickup_bk = (((X['pickup_latitude'] - 40.6928) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.9872) * 84.0) ** 2) ** 0.5
    dist_dropoff_bk = (((X['dropoff_latitude'] - 40.6928) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.9872) * 84.0) ** 2) ** 0.5
    dist_pickup_uptown = (((X['pickup_latitude'] - 40.8) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.958) * 84.0) ** 2) ** 0.5
    dist_dropoff_uptown = (((X['dropoff_latitude'] - 40.8) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.958) * 84.0) ** 2) ** 0.5
    dist_pickup_queens = (((X['pickup_latitude'] - 40.7505) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.937) * 84.0) ** 2) ** 0.5
    dist_dropoff_queens = (((X['dropoff_latitude'] - 40.7505) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.937) * 84.0) ** 2) ** 0.5
    pickup_is_manhattan = ((X['pickup_longitude'] >= -74.02) & (X['pickup_longitude'] <= -73.93) & (X['pickup_latitude'] >= 40.7) & (X['pickup_latitude'] <= 40.85)).astype('float32')
    dropoff_is_manhattan = ((X['dropoff_longitude'] >= -74.02) & (X['dropoff_longitude'] <= -73.93) & (X['dropoff_latitude'] >= 40.7) & (X['dropoff_latitude'] <= 40.85)).astype('float32')
    cross_borough_east = ((pickup_is_manhattan == 1) & (X['dropoff_longitude'] > -73.93) | (dropoff_is_manhattan == 1) & (X['pickup_longitude'] > -73.93)).astype('float32')
    mta_tax = ((year > 2009) | (year == 2009) & (month >= 11)).astype('float32')
    morning_rush = ((weekday < 5) & (hour >= 7) & (hour < 10)).astype('float32')
    fare_hike_2012 = ((year > 2012) | (year == 2012) & (month >= 9)).astype('float32')
    rush_hour_surcharge = ((weekday < 5) & (hour >= 16) & (hour < 20)).astype('float32')
    overnight_surcharge = ((hour >= 20) | (hour < 6)).astype('float32')
    dist_x_rush = (rotated_manhattan_dist * (rush_hour_surcharge + morning_rush)).astype('float32')
    hour_rad = hour * (2.0 * np.pi / 24.0)
    hour_sin = hour_rad.skb.apply_func(np.sin).astype('float32')
    hour_cos = hour_rad.skb.apply_func(np.cos).astype('float32')
    jfk_trip = ((dist_pickup_jfk < 3.0) | (dist_dropoff_jfk < 3.0)).astype('float32')
    lga_trip = ((dist_pickup_lga < 3.0) | (dist_dropoff_lga < 3.0)).astype('float32')
    ewr_trip = ((dist_pickup_ewr < 3.0) | (dist_dropoff_ewr < 3.0)).astype('float32')
    jfk_manhattan_flat = ((dist_pickup_jfk < 3.0) & (dist_dropoff_midtown < 5.0) | (dist_dropoff_jfk < 3.0) & (dist_pickup_midtown < 5.0)).astype('float32')
    jfk_flat_post_2012 = (jfk_manhattan_flat * fare_hike_2012).astype('float32')
    jfk_statutory_fare = (jfk_manhattan_flat * (45.0 + 7.0 * fare_hike_2012)).astype('float32')
    ewr_statutory_surcharge = (ewr_trip * (15.0 + 2.5 * fare_hike_2012)).astype('float32')
    dist_x_fare_hike = (rotated_manhattan_dist * fare_hike_2012).astype('float32')
    is_stationary = (haversine_dist < 0.05).astype('float32')
    features_base = X.drop(columns=['key', 'pickup_datetime']).assign(abs_diff_lon=d_lon, abs_diff_lat=d_lat, euclidean_dist=euclidean_dist, haversine_dist=haversine_dist, rotated_manhattan_dist=rotated_manhattan_dist, avenue_dist=avenue_dist, street_dist=street_dist, mid_longitude=mid_longitude, mid_latitude=mid_latitude, fractional_year=fractional_year, bearing=bearing, dist_pickup_jfk=dist_pickup_jfk, dist_dropoff_jfk=dist_dropoff_jfk, dist_pickup_lga=dist_pickup_lga, dist_dropoff_lga=dist_dropoff_lga, dist_pickup_ewr=dist_pickup_ewr, dist_dropoff_ewr=dist_dropoff_ewr, dist_pickup_midtown=dist_pickup_midtown, dist_dropoff_midtown=dist_dropoff_midtown, dist_pickup_downtown=dist_pickup_downtown, dist_dropoff_downtown=dist_dropoff_downtown, year=year, month=month, day=day, hour=hour, weekday=weekday, hour_sin=hour_sin, hour_cos=hour_cos, fare_hike_2012=fare_hike_2012, rush_hour_surcharge=rush_hour_surcharge, overnight_surcharge=overnight_surcharge, jfk_trip=jfk_trip, lga_trip=lga_trip, ewr_trip=ewr_trip, jfk_manhattan_flat=jfk_manhattan_flat, jfk_flat_post_2012=jfk_flat_post_2012, dist_x_fare_hike=dist_x_fare_hike, is_stationary=is_stationary)
    features_v1 = features_base.assign(dist_pickup_bk=dist_pickup_bk, dist_dropoff_bk=dist_dropoff_bk, dist_pickup_uptown=dist_pickup_uptown, dist_dropoff_uptown=dist_dropoff_uptown, cross_borough_east=cross_borough_east, mta_tax=mta_tax, morning_rush=morning_rush, dist_x_rush=dist_x_rush)
    features_v2 = features_v1.assign(dist_pickup_queens=dist_pickup_queens, dist_dropoff_queens=dist_dropoff_queens, jfk_statutory_fare=jfk_statutory_fare, ewr_statutory_surcharge=ewr_statutory_surcharge)
    p_in_cp_lat = (X['pickup_latitude'] >= 40.764) & (X['pickup_latitude'] <= 40.805)
    d_in_cp_lat = (X['dropoff_latitude'] >= 40.764) & (X['dropoff_latitude'] <= 40.805)
    p_west = X['pickup_longitude'] < -73.972
    p_east = X['pickup_longitude'] > -73.958
    d_west = X['dropoff_longitude'] < -73.972
    d_east = X['dropoff_longitude'] > -73.958
    cross_central_park = (p_in_cp_lat & d_in_cp_lat & (p_west & d_east | p_east & d_west)).astype('float32')
    cross_hudson = ((X['pickup_longitude'] > -74.01) & (X['dropoff_longitude'] < -74.035) | (X['pickup_longitude'] < -74.035) & (X['dropoff_longitude'] > -74.01)).astype('float32')
    tortuosity = (rotated_manhattan_dist / (euclidean_dist + 0.1)).astype('float32')
    log_haversine = haversine_dist.skb.apply_func(np.log1p).astype('float32')
    dist_pickup_bronx = (((X['pickup_latitude'] - 40.8448) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.8648) * 84.0) ** 2) ** 0.5
    dist_dropoff_bronx = (((X['dropoff_latitude'] - 40.8448) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.8648) * 84.0) ** 2) ** 0.5
    bronx_trip = ((dist_pickup_bronx < 4.0) | (dist_dropoff_bronx < 4.0)).astype('float32')
    features_corridor_bronx = features_v2.assign(cross_central_park=cross_central_park, cross_hudson=cross_hudson, tortuosity=tortuosity, log_haversine=log_haversine, dist_pickup_bronx=dist_pickup_bronx, dist_dropoff_bronx=dist_dropoff_bronx, bronx_trip=bronx_trip)
    jfk_rush_surcharge = (jfk_manhattan_flat * rush_hour_surcharge * 4.5).astype('float32')
    is_weekend = (weekday >= 5).astype('float32')
    features_peak_tariffs = features_corridor_bronx.assign(jfk_rush_surcharge=jfk_rush_surcharge, is_weekend=is_weekend)
    features = skrub.as_data_op(features_peak_tariffs).skb.set_name('features')
    lgb_est = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=127, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb_est = xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42, tree_method='hist', device='cuda', n_jobs=-1)
    mlp_est = TorchTabularRegressor()
    weights_choice = skrub.choose_from({'balanced_trees_44_44_12': [0.44, 0.44, 0.12], 'xgb_dominant_43_45_12': [0.43, 0.45, 0.12]}, name='blend_weights')
    model = VotingRegressor(estimators=[('lgb', lgb_est), ('xgb', xgb_est), ('mlp', mlp_est)], weights=weights_choice)
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}