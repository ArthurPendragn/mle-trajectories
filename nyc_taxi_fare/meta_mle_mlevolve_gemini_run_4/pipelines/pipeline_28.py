import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import torch

class TaxiFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        p_lon = X['pickup_longitude'].to_numpy(dtype=np.float64)
        p_lat = X['pickup_latitude'].to_numpy(dtype=np.float64)
        d_lon = X['dropoff_longitude'].to_numpy(dtype=np.float64)
        d_lat = X['dropoff_latitude'].to_numpy(dtype=np.float64)
        passengers = X['passenger_count'].to_numpy(dtype=np.float64)
        dlon = d_lon - p_lon
        dlat = d_lat - p_lat
        abs_dlon = np.abs(dlon)
        abs_dlat = np.abs(dlat)
        lat1, lon1 = (np.radians(p_lat), np.radians(p_lon))
        lat2, lon2 = (np.radians(d_lat), np.radians(d_lon))
        h_dlat = lat2 - lat1
        h_dlon = lon2 - lon1
        a = np.sin(h_dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(h_dlon / 2.0) ** 2
        haversine_km = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        rot_rad = 29.0 * np.pi / 180.0
        cos_rot = np.cos(rot_rad)
        sin_rot = np.sin(rot_rad)
        mean_lat = np.radians(40.75)
        dx_km = dlon * 111.32 * np.cos(mean_lat)
        dy_km = dlat * 110.57
        dx_rot = dx_km * cos_rot - dy_km * sin_rot
        dy_rot = dx_km * sin_rot + dy_km * cos_rot
        rot_dx = np.abs(dx_rot)
        rot_dy = np.abs(dy_rot)
        manhattan_km = rot_dx + rot_dy
        euclidean_km = np.sqrt(dx_km ** 2 + dy_km ** 2)
        jfk_p_dist = np.sqrt((p_lon - -73.7781) ** 2 + (p_lat - 40.6413) ** 2)
        jfk_d_dist = np.sqrt((d_lon - -73.7781) ** 2 + (d_lat - 40.6413) ** 2)
        is_jfk = ((jfk_p_dist < 0.03) | (jfk_d_dist < 0.03)).astype(np.float64)
        lga_p_dist = np.sqrt((p_lon - -73.874) ** 2 + (p_lat - 40.7769) ** 2)
        lga_d_dist = np.sqrt((d_lon - -73.874) ** 2 + (d_lat - 40.7769) ** 2)
        is_lga = ((lga_p_dist < 0.03) | (lga_d_dist < 0.03)).astype(np.float64)
        ewr_p_dist = np.sqrt((p_lon - -74.1745) ** 2 + (p_lat - 40.6895) ** 2)
        ewr_d_dist = np.sqrt((d_lon - -74.1745) ** 2 + (d_lat - 40.6895) ** 2)
        is_ewr = ((ewr_p_dist < 0.03) | (ewr_d_dist < 0.03)).astype(np.float64)
        p_manhattan = ((p_lon >= -74.02) & (p_lon <= -73.93) & (p_lat >= 40.7) & (p_lat <= 40.87)).astype(np.float64)
        d_manhattan = ((d_lon >= -74.02) & (d_lon <= -73.93) & (d_lat >= 40.7) & (d_lat <= 40.87)).astype(np.float64)
        is_both_manhattan = p_manhattan * d_manhattan
        is_cross_borough = (p_manhattan != d_manhattan).astype(np.float64)
        p_brooklyn = ((p_lon >= -74.04) & (p_lon <= -73.85) & (p_lat >= 40.57) & (p_lat <= 40.73)).astype(np.float64)
        d_brooklyn = ((d_lon >= -74.04) & (d_lon <= -73.85) & (d_lat >= 40.57) & (d_lat <= 40.73)).astype(np.float64)
        p_queens = ((p_lon >= -73.96) & (p_lon <= -73.7) & (p_lat >= 40.54) & (p_lat <= 40.8)).astype(np.float64)
        d_queens = ((d_lon >= -73.96) & (d_lon <= -73.7) & (d_lat >= 40.54) & (d_lat <= 40.8)).astype(np.float64)
        p_bronx = ((p_lon >= -73.93) & (p_lon <= -73.76) & (p_lat >= 40.8) & (p_lat <= 40.91)).astype(np.float64)
        d_bronx = ((d_lon >= -73.93) & (d_lon <= -73.76) & (d_lat >= 40.8) & (d_lat <= 40.91)).astype(np.float64)
        is_manhattan_brooklyn = ((p_manhattan > 0) & (d_brooklyn > 0) | (d_manhattan > 0) & (p_brooklyn > 0)).astype(np.float64)
        is_manhattan_queens = ((p_manhattan > 0) & (d_queens > 0) | (d_manhattan > 0) & (p_queens > 0)).astype(np.float64)
        is_manhattan_bronx = ((p_manhattan > 0) & (d_bronx > 0) | (d_manhattan > 0) & (p_bronx > 0)).astype(np.float64)
        is_jfk_manhattan = ((jfk_p_dist < 0.03) & (d_manhattan > 0) | (jfk_d_dist < 0.03) & (p_manhattan > 0)).astype(np.float64)
        is_ewr_manhattan = ((ewr_p_dist < 0.03) & (d_manhattan > 0) | (ewr_d_dist < 0.03) & (p_manhattan > 0)).astype(np.float64)
        center_lat = (p_lat + d_lat) / 2.0
        center_lon = (p_lon + d_lon) / 2.0
        mid_midtown_tunnel = np.sqrt(((center_lat - 40.744) * 111.13) ** 2 + ((center_lon - -73.968) * 84.18) ** 2)
        mid_rfk = np.sqrt(((center_lat - 40.78) * 111.13) ** 2 + ((center_lon - -73.924) * 84.18) ** 2)
        mid_queensboro = np.sqrt(((center_lat - 40.757) * 111.13) ** 2 + ((center_lon - -73.955) * 84.18) ** 2)
        mid_battery_tunnel = np.sqrt(((center_lat - 40.697) * 111.13) ** 2 + ((center_lon - -74.015) * 84.18) ** 2)
        mid_brooklyn_bridge = np.sqrt(((center_lat - 40.706) * 111.13) ** 2 + ((center_lon - -73.997) * 84.18) ** 2)
        mid_lincoln = np.sqrt(((center_lat - 40.76) * 111.13) ** 2 + ((center_lon - -74.004) * 84.18) ** 2)
        mid_holland = np.sqrt(((center_lat - 40.726) * 111.13) ** 2 + ((center_lon - -74.016) * 84.18) ** 2)
        mid_gwb = np.sqrt(((center_lat - 40.852) * 111.13) ** 2 + ((center_lon - -73.953) * 84.18) ** 2)
        tolled_midtown_tunnel = ((is_manhattan_queens > 0) & (mid_midtown_tunnel < 3.0)).astype(np.float64)
        tolled_rfk = (((is_manhattan_queens > 0) | (is_manhattan_bronx > 0)) & (mid_rfk < 3.0)).astype(np.float64)
        tolled_battery_tunnel = ((is_manhattan_brooklyn > 0) & (mid_battery_tunnel < 2.5)).astype(np.float64)
        tolled_hudson_tunnel = ((is_ewr_manhattan > 0) | (d_lon < -74.03)).astype(np.float64)
        midtown_p_dist = np.sqrt(((p_lat - 40.755) * 111.13) ** 2 + ((p_lon - -73.985) * 84.18) ** 2)
        midtown_d_dist = np.sqrt(((d_lat - 40.755) * 111.13) ** 2 + ((d_lon - -73.985) * 84.18) ** 2)
        fidi_p_dist = np.sqrt(((p_lat - 40.7128) * 111.13) ** 2 + ((p_lon - -74.006) * 84.18) ** 2)
        fidi_d_dist = np.sqrt(((d_lat - 40.7128) * 111.13) ** 2 + ((d_lon - -74.006) * 84.18) ** 2)
        is_zero_dist = (manhattan_km < 0.05).astype(np.float64)
        dt = pd.to_datetime(X['pickup_datetime'].astype(str).str.slice(0, 19), format='%Y-%m-%d %H:%M:%S')
        year = dt.dt.year.to_numpy(dtype=np.float64)
        month = dt.dt.month.to_numpy(dtype=np.float64)
        day = dt.dt.day.to_numpy(dtype=np.float64)
        hour = dt.dt.hour.to_numpy(dtype=np.float64)
        weekday = dt.dt.weekday.to_numpy(dtype=np.float64)
        sin_hour = np.sin(2.0 * np.pi * hour / 24.0)
        cos_hour = np.cos(2.0 * np.pi * hour / 24.0)
        sin_month = np.sin(2.0 * np.pi * month / 12.0)
        cos_month = np.cos(2.0 * np.pi * month / 12.0)
        is_weekend = (weekday >= 5).astype(np.float64)
        is_rush_hour = ((weekday < 5) & (hour >= 16) & (hour < 20)).astype(np.float64)
        is_overnight = ((hour >= 20) | (hour < 6)).astype(np.float64)
        is_post_2012 = ((year > 2012) | (year == 2012) & ((month > 9) | (month == 9) & (day >= 4))).astype(np.float64)
        is_post_nov_2009 = ((year > 2009) | (year == 2009) & (month >= 11)).astype(np.float64)
        non_jfk_manhattan_km = (1.0 - is_jfk_manhattan) * manhattan_km
        non_jfk_haversine_km = (1.0 - is_jfk_manhattan) * haversine_km
        jfk_flat_rate = is_jfk_manhattan * (52.0 * is_post_2012 + 45.0 * (1.0 - is_post_2012))
        ewr_surcharge = is_ewr_manhattan * (17.5 * is_post_2012 + 15.0 * (1.0 - is_post_2012))
        statutory_surcharge = 0.5 * is_post_nov_2009 + 0.5 * is_overnight + 1.0 * is_rush_hour
        post_2012_dist = is_post_2012 * non_jfk_manhattan_km
        post_2012_hav = is_post_2012 * non_jfk_haversine_km
        sqrt_manhattan_km = np.sqrt(non_jfk_manhattan_km)
        log_manhattan_km = np.log1p(non_jfk_manhattan_km)
        sqrt_haversine_km = np.sqrt(non_jfk_haversine_km)
        log_haversine_km = np.log1p(non_jfk_haversine_km)
        rush_hour_dist = is_rush_hour * non_jfk_manhattan_km
        overnight_dist = is_overnight * non_jfk_manhattan_km
        post_2012_rush = is_post_2012 * is_rush_hour
        rate_per_km = 1.2427 * (1.0 - is_post_2012) + 1.5534 * is_post_2012
        effective_dist_km = is_both_manhattan * manhattan_km + (1.0 - is_both_manhattan) * (0.6 * manhattan_km + 0.4 * haversine_km)
        metered_dist_fare = effective_dist_km * rate_per_km
        base_fare = 2.5 + metered_dist_fare
        statutory_fare_estimate = (1.0 - is_jfk_manhattan) * base_fare + jfk_flat_rate + statutory_surcharge + ewr_surcharge
        feats = {'pickup_longitude': p_lon, 'pickup_latitude': p_lat, 'dropoff_longitude': d_lon, 'dropoff_latitude': d_lat, 'passenger_count': passengers, 'dlon': dlon, 'dlat': dlat, 'abs_dlon': abs_dlon, 'abs_dlat': abs_dlat, 'haversine_km': haversine_km, 'manhattan_km': manhattan_km, 'euclidean_km': euclidean_km, 'non_jfk_manhattan_km': non_jfk_manhattan_km, 'non_jfk_haversine_km': non_jfk_haversine_km, 'post_2012_dist': post_2012_dist, 'post_2012_hav': post_2012_hav, 'sqrt_manhattan_km': sqrt_manhattan_km, 'log_manhattan_km': log_manhattan_km, 'sqrt_haversine_km': sqrt_haversine_km, 'log_haversine_km': log_haversine_km, 'rush_hour_dist': rush_hour_dist, 'overnight_dist': overnight_dist, 'post_2012_rush': post_2012_rush, 'jfk_flat_rate': jfk_flat_rate, 'ewr_surcharge': ewr_surcharge, 'statutory_surcharge': statutory_surcharge, 'statutory_fare_estimate': statutory_fare_estimate, 'jfk_p_dist': jfk_p_dist, 'jfk_d_dist': jfk_d_dist, 'is_jfk': is_jfk, 'is_jfk_manhattan': is_jfk_manhattan, 'lga_p_dist': lga_p_dist, 'lga_d_dist': lga_d_dist, 'is_lga': is_lga, 'ewr_p_dist': ewr_p_dist, 'ewr_d_dist': ewr_d_dist, 'is_ewr': is_ewr, 'is_ewr_manhattan': is_ewr_manhattan, 'p_manhattan': p_manhattan, 'd_manhattan': d_manhattan, 'is_both_manhattan': is_both_manhattan, 'is_cross_borough': is_cross_borough, 'is_manhattan_brooklyn': is_manhattan_brooklyn, 'is_manhattan_queens': is_manhattan_queens, 'is_manhattan_bronx': is_manhattan_bronx, 'mid_midtown_tunnel': mid_midtown_tunnel, 'mid_rfk': mid_rfk, 'mid_queensboro': mid_queensboro, 'mid_battery_tunnel': mid_battery_tunnel, 'mid_brooklyn_bridge': mid_brooklyn_bridge, 'mid_lincoln': mid_lincoln, 'mid_holland': mid_holland, 'mid_gwb': mid_gwb, 'tolled_midtown_tunnel': tolled_midtown_tunnel, 'tolled_rfk': tolled_rfk, 'tolled_battery_tunnel': tolled_battery_tunnel, 'tolled_hudson_tunnel': tolled_hudson_tunnel, 'midtown_p_dist': midtown_p_dist, 'midtown_d_dist': midtown_d_dist, 'fidi_p_dist': fidi_p_dist, 'fidi_d_dist': fidi_d_dist, 'is_zero_dist': is_zero_dist, 'year': year, 'sin_hour': sin_hour, 'cos_hour': cos_hour, 'sin_month': sin_month, 'cos_month': cos_month, 'is_weekend': is_weekend, 'is_rush_hour': is_rush_hour, 'is_overnight': is_overnight, 'is_post_2012': is_post_2012}
        return pd.DataFrame(feats)

class DualBoosterResidualRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        lgb_n_estimators: int = 1000,
        lgb_learning_rate: float = 0.04,
        lgb_num_leaves: int = 63,
        lgb_min_child_samples: int = 50,
        lgb_reg_alpha: float = 1.0,
        lgb_reg_lambda: float = 5.0,
        lgb_colsample_bytree: float = 0.8,
        lgb_subsample: float = 0.8,
        lgb_subsample_freq: int = 1,
        xgb_n_estimators: int = 1000,
        xgb_learning_rate: float = 0.04,
        xgb_max_depth: int = 8,
        xgb_reg_alpha: float = 1.0,
        xgb_reg_lambda: float = 5.0,
        xgb_colsample_bytree: float = 0.8,
        xgb_subsample: float = 0.8,
        w_lgb: float = 0.5,
        w_xgb: float = 0.5,
        fare_floor: float = 2.5,
        fare_cap: float = 500.0,
        random_state: int = 42,
        n_jobs: int = 8,
    ):
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
        self.w_lgb = w_lgb
        self.w_xgb = w_xgb
        self.fare_floor = fare_floor
        self.fare_cap = fare_cap
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            statutory_baseline = X['statutory_fare_estimate'].to_numpy(dtype=np.float64)
            self.baseline_idx_ = X.columns.get_loc('statutory_fare_estimate')
        else:
            self.baseline_idx_ = -1
            statutory_baseline = X[:, self.baseline_idx_].astype(np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        residual = y_arr - statutory_baseline
        X_mat = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        self.lgb_ = LGBMRegressor(
            n_estimators=self.lgb_n_estimators,
            learning_rate=self.lgb_learning_rate,
            num_leaves=self.lgb_num_leaves,
            min_child_samples=self.lgb_min_child_samples,
            reg_alpha=self.lgb_reg_alpha,
            reg_lambda=self.lgb_reg_lambda,
            colsample_bytree=self.lgb_colsample_bytree,
            subsample=self.lgb_subsample,
            subsample_freq=self.lgb_subsample_freq,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self.lgb_.fit(X, residual)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.xgb_ = XGBRegressor(
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                max_depth=self.xgb_max_depth,
                reg_alpha=self.xgb_reg_alpha,
                reg_lambda=self.xgb_reg_lambda,
                colsample_bytree=self.xgb_colsample_bytree,
                subsample=self.xgb_subsample,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tree_method='hist',
                device=device,
            )
            self.xgb_.fit(X_mat, residual)
        except Exception:
            self.xgb_ = XGBRegressor(
                n_estimators=self.xgb_n_estimators,
                learning_rate=self.xgb_learning_rate,
                max_depth=self.xgb_max_depth,
                reg_alpha=self.xgb_reg_alpha,
                reg_lambda=self.xgb_reg_lambda,
                colsample_bytree=self.xgb_colsample_bytree,
                subsample=self.xgb_subsample,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                tree_method='hist',
            )
            self.xgb_.fit(X_mat, residual)

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            statutory_baseline = X['statutory_fare_estimate'].to_numpy(dtype=np.float64)
        else:
            statutory_baseline = X[:, self.baseline_idx_].astype(np.float64)
        X_mat = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        pred_lgb = self.lgb_.predict(X)
        pred_xgb = self.xgb_.predict(X_mat)
        pred_res = self.w_lgb * pred_lgb + self.w_xgb * pred_xgb
        pred = statutory_baseline + pred_res
        return np.clip(pred, self.fare_floor, self.fare_cap)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']
    features = X.skb.apply(TaxiFeatureTransformer()).skb.set_name('features')
    pred = features.skb.apply(DualBoosterResidualRegressor(), y=y).skb.set_name('model')
    result = {'pred': pred, 'scoring': scoring}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result