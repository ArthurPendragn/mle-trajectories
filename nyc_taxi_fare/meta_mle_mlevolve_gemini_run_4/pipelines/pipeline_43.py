import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor

class TaxiDomainFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        dt = pd.to_datetime(df['pickup_datetime'], errors='coerce', utc=True)
        p_lat = df['pickup_latitude'].to_numpy(dtype=np.float64)
        p_lon = df['pickup_longitude'].to_numpy(dtype=np.float64)
        d_lat = df['dropoff_latitude'].to_numpy(dtype=np.float64)
        d_lon = df['dropoff_longitude'].to_numpy(dtype=np.float64)
        passengers = df['passenger_count'].to_numpy(dtype=np.float64)

        dlat = d_lat - p_lat
        dlon = d_lon - p_lon
        abs_lat_diff = np.abs(dlat)
        abs_lon_diff = np.abs(dlon)

        rad_dlat = np.radians(dlat)
        rad_dlon = np.radians(dlon)
        lat1 = np.radians(p_lat)
        lat2 = np.radians(d_lat)
        a = np.sin(rad_dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(rad_dlon / 2.0) ** 2
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arcsin(np.sqrt(a))
        haversine_km = 6371.0 * c

        dlat_km = abs_lat_diff * 111.0
        dlon_km = abs_lon_diff * (111.0 * np.cos(np.radians(40.75)))
        manhattan_km = dlat_km + dlon_km

        rot_rad = 29.0 * np.pi / 180.0
        cos_rot = np.cos(rot_rad)
        sin_rot = np.sin(rot_rad)
        mean_lat = np.radians(40.75)
        dx_km = dlon * 111.32 * np.cos(mean_lat)
        dy_km = dlat * 110.57
        dx_rot = dx_km * cos_rot - dy_km * sin_rot
        dy_rot = dx_km * sin_rot + dy_km * cos_rot
        rot_dlon_km = np.abs(dx_rot)
        rot_dlat_km = np.abs(dy_rot)
        rot_manhattan_km = rot_dlon_km + rot_dlat_km

        jfk_p_dist = np.sqrt((p_lon - -73.7781) ** 2 + (p_lat - 40.6413) ** 2)
        jfk_d_dist = np.sqrt((d_lon - -73.7781) ** 2 + (d_lat - 40.6413) ** 2)
        lga_p_dist = np.sqrt((p_lon - -73.874) ** 2 + (p_lat - 40.7769) ** 2)
        lga_d_dist = np.sqrt((d_lon - -73.874) ** 2 + (d_lat - 40.7769) ** 2)
        ewr_p_dist = np.sqrt((p_lon - -74.1745) ** 2 + (p_lat - 40.6895) ** 2)
        ewr_d_dist = np.sqrt((d_lon - -74.1745) ** 2 + (d_lat - 40.6895) ** 2)

        is_jfk = ((jfk_p_dist < 0.03) | (jfk_d_dist < 0.03)).astype(np.float64)
        is_lga = ((lga_p_dist < 0.03) | (lga_d_dist < 0.03)).astype(np.float64)
        is_ewr = ((ewr_p_dist < 0.03) | (ewr_d_dist < 0.03)).astype(np.float64)

        hour = dt.dt.hour.fillna(12).to_numpy(dtype=np.float64)
        dayofweek = dt.dt.dayofweek.fillna(2).to_numpy(dtype=np.float64)
        month = dt.dt.month.fillna(6).to_numpy(dtype=np.float64)
        day = dt.dt.day.fillna(15).to_numpy(dtype=np.float64)
        year = dt.dt.year.fillna(2012).to_numpy(dtype=np.float64)

        is_post_2012 = ((year > 2012) | ((year == 2012) & ((month > 9) | ((month == 9) & (day >= 4))))).astype(np.float64)
        is_post_nov_2009 = ((year > 2009) | ((year == 2009) & (month >= 11))).astype(np.float64)
        is_rush_hour = ((dayofweek < 5) & (hour >= 16) & (hour < 20)).astype(np.float64)
        is_overnight = ((hour >= 20) | (hour < 6)).astype(np.float64)

        statutory_surcharge = 0.5 * is_post_nov_2009 + 0.5 * is_overnight + 1.0 * is_rush_hour
        rate_per_km = 1.2427 * (1.0 - is_post_2012) + 1.5534 * is_post_2012
        metered_fare = 2.5 + rot_manhattan_km * rate_per_km
        jfk_flat = 45.0 * (1.0 - is_post_2012) + 52.0 * is_post_2012
        statutory_fare_estimate = np.where(is_jfk > 0.5, jfk_flat, metered_fare) + statutory_surcharge

        features = pd.DataFrame(
            {
                'pickup_longitude': p_lon,
                'pickup_latitude': p_lat,
                'dropoff_longitude': d_lon,
                'dropoff_latitude': d_lat,
                'passenger_count': passengers,
                'haversine_km': haversine_km,
                'manhattan_km': manhattan_km,
                'abs_lat_diff': abs_lat_diff,
                'abs_lon_diff': abs_lon_diff,
                'center_latitude': (p_lat + d_lat) / 2.0,
                'center_longitude': (p_lon + d_lon) / 2.0,
                'rot_dlon_km': rot_dlon_km,
                'rot_dlat_km': rot_dlat_km,
                'rot_manhattan_km': rot_manhattan_km,
                'jfk_p_dist': jfk_p_dist,
                'jfk_d_dist': jfk_d_dist,
                'lga_p_dist': lga_p_dist,
                'lga_d_dist': lga_d_dist,
                'ewr_p_dist': ewr_p_dist,
                'ewr_d_dist': ewr_d_dist,
                'is_jfk': is_jfk,
                'is_lga': is_lga,
                'is_ewr': is_ewr,
                'statutory_fare_estimate': statutory_fare_estimate,
                'hour': hour,
                'dayofweek': dayofweek,
                'month': month,
                'year': year,
            },
            index=df.index,
        )
        return features

class ResidualForestRegressor(RegressorMixin, BaseEstimator):

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 16,
        max_samples: float = 0.3,
        fare_floor: float = 2.5,
        fare_cap: float = 500.0,
        n_jobs: int = 8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.fare_floor = fare_floor
        self.fare_cap = fare_cap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            statutory_baseline = X['statutory_fare_estimate'].to_numpy(dtype=np.float64)
            self.baseline_col_ = 'statutory_fare_estimate'
        else:
            statutory_baseline = X[:, -1].astype(np.float64)
            self.baseline_col_ = None

        y_arr = np.asarray(y, dtype=np.float64)
        residual = y_arr - statutory_baseline

        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_samples=self.max_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.rf_.fit(X, residual)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            statutory_baseline = X['statutory_fare_estimate'].to_numpy(dtype=np.float64)
        else:
            statutory_baseline = X[:, -1].astype(np.float64)

        pred_residual = self.rf_.predict(X)
        pred = statutory_baseline + pred_residual
        return np.clip(pred, self.fare_floor, self.fare_cap)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    features = X.skb.apply(TaxiDomainFeatureExtractor()).skb.set_name('features')
    model = ResidualForestRegressor(
        n_estimators=60,
        max_depth=16,
        max_samples=0.3,
        n_jobs=8,
        random_state=42,
    )
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    result = {'pred': pred, 'scoring': setup['scoring']}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result