import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

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

        lat1, lon1 = np.radians(p_lat), np.radians(p_lon)
        lat2, lon2 = np.radians(d_lat), np.radians(d_lon)
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

        p_manhattan = ((p_lon >= -74.02) & (p_lon <= -73.93) & (p_lat >= 40.70) & (p_lat <= 40.87)).astype(np.float64)
        d_manhattan = ((d_lon >= -74.02) & (d_lon <= -73.93) & (d_lat >= 40.70) & (d_lat <= 40.87)).astype(np.float64)
        is_jfk_manhattan = (((jfk_p_dist < 0.03) & (d_manhattan > 0)) | ((jfk_d_dist < 0.03) & (p_manhattan > 0))).astype(np.float64)
        is_ewr_manhattan = (((ewr_p_dist < 0.03) & (d_manhattan > 0)) | ((ewr_d_dist < 0.03) & (p_manhattan > 0))).astype(np.float64)

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
        is_post_2012 = ((year > 2012) | ((year == 2012) & ((month > 9) | ((month == 9) & (day >= 4))))).astype(np.float64)
        is_post_nov_2009 = ((year > 2009) | ((year == 2009) & (month >= 11))).astype(np.float64)

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

        feats = {
            'pickup_longitude': p_lon,
            'pickup_latitude': p_lat,
            'dropoff_longitude': d_lon,
            'dropoff_latitude': d_lat,
            'passenger_count': passengers,
            'dlon': dlon,
            'dlat': dlat,
            'abs_dlon': abs_dlon,
            'abs_dlat': abs_dlat,
            'haversine_km': haversine_km,
            'manhattan_km': manhattan_km,
            'euclidean_km': euclidean_km,
            'non_jfk_manhattan_km': non_jfk_manhattan_km,
            'non_jfk_haversine_km': non_jfk_haversine_km,
            'post_2012_dist': post_2012_dist,
            'post_2012_hav': post_2012_hav,
            'sqrt_manhattan_km': sqrt_manhattan_km,
            'log_manhattan_km': log_manhattan_km,
            'sqrt_haversine_km': sqrt_haversine_km,
            'log_haversine_km': log_haversine_km,
            'rush_hour_dist': rush_hour_dist,
            'overnight_dist': overnight_dist,
            'post_2012_rush': post_2012_rush,
            'jfk_flat_rate': jfk_flat_rate,
            'ewr_surcharge': ewr_surcharge,
            'statutory_surcharge': statutory_surcharge,
            'jfk_p_dist': jfk_p_dist,
            'jfk_d_dist': jfk_d_dist,
            'is_jfk': is_jfk,
            'is_jfk_manhattan': is_jfk_manhattan,
            'lga_p_dist': lga_p_dist,
            'lga_d_dist': lga_d_dist,
            'is_lga': is_lga,
            'ewr_p_dist': ewr_p_dist,
            'ewr_d_dist': ewr_d_dist,
            'is_ewr': is_ewr,
            'is_ewr_manhattan': is_ewr_manhattan,
            'year': year,
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'sin_month': sin_month,
            'cos_month': cos_month,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour,
            'is_overnight': is_overnight,
            'is_post_2012': is_post_2012,
        }
        return pd.DataFrame(feats)

class ClippedRidgeRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.alphas = alphas

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = RidgeCV(alphas=self.alphas)
        y_arr = np.asarray(y, dtype=np.float64)
        self.model_.fit(X_scaled, y_arr)
        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        preds = self.model_.predict(X_scaled)
        return np.clip(preds, 2.5, 500.0)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']
    features = X.skb.apply(TaxiFeatureTransformer()).skb.set_name('features')
    pred = features.skb.apply(ClippedRidgeRegressor(), y=y).skb.set_name('model')
    result = {'pred': pred, 'scoring': scoring}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result