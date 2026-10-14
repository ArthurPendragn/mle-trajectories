import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import skrub


class TaxiFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def _haversine(self, lat1, lon1, lat2, lon2):
        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlat_r = np.radians(lat2 - lat1)
        dlon_r = np.radians(lon2 - lon1)
        a = np.sin(dlat_r / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon_r / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return 6371.0 * c

    def fit(self, X, y=None):
        self.median_p_lat_ = float(X['pickup_latitude'].median())
        self.median_p_lon_ = float(X['pickup_longitude'].median())
        self.median_d_lat_ = float(X['dropoff_latitude'].median())
        self.median_d_lon_ = float(X['dropoff_longitude'].median())
        return self

    def transform(self, X):
        p_lat = X['pickup_latitude'].fillna(self.median_p_lat_).astype('float64')
        p_lon = X['pickup_longitude'].fillna(self.median_p_lon_).astype('float64')
        d_lat = X['dropoff_latitude'].fillna(self.median_d_lat_).astype('float64')
        d_lon = X['dropoff_longitude'].fillna(self.median_d_lon_).astype('float64')
        passengers = X['passenger_count'].fillna(1).astype('int32')

        dt = pd.to_datetime(X['pickup_datetime'])
        year = dt.dt.year.astype('int32')
        month = dt.dt.month.astype('int32')
        day = dt.dt.day.astype('int32')
        dayofweek = dt.dt.dayofweek.astype('int32')
        hour = dt.dt.hour.astype('int32')
        is_weekend = (dayofweek >= 5).astype('int32')

        lat_diff = (d_lat - p_lat).abs()
        lon_diff = (d_lon - p_lon).abs()
        haversine_dist = self._haversine(p_lat, p_lon, d_lat, d_lon)
        manhattan_dist = self._haversine(p_lat, p_lon, p_lat, d_lon) + self._haversine(p_lat, d_lon, d_lat, d_lon)

        lat1_r = np.radians(p_lat)
        lat2_r = np.radians(d_lat)
        dlon_r = np.radians(d_lon - p_lon)
        bearing = np.arctan2(
            np.sin(dlon_r) * np.cos(lat2_r),
            np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon_r)
        )

        jfk_p = self._haversine(p_lat, p_lon, 40.6413, -73.7781)
        jfk_d = self._haversine(d_lat, d_lon, 40.6413, -73.7781)
        lga_p = self._haversine(p_lat, p_lon, 40.7769, -73.8740)
        lga_d = self._haversine(d_lat, d_lon, 40.7769, -73.8740)
        ewr_p = self._haversine(p_lat, p_lon, 40.6895, -74.1745)
        ewr_d = self._haversine(d_lat, d_lon, 40.6895, -74.1745)
        mid_p = self._haversine(p_lat, p_lon, 40.7580, -73.9855)
        mid_d = self._haversine(d_lat, d_lon, 40.7580, -73.9855)

        features = pd.DataFrame(
            {
                'pickup_latitude': p_lat,
                'pickup_longitude': p_lon,
                'dropoff_latitude': d_lat,
                'dropoff_longitude': d_lon,
                'passenger_count': passengers,
                'year': year,
                'month': month,
                'day': day,
                'dayofweek': dayofweek,
                'hour': hour,
                'is_weekend': is_weekend,
                'lat_diff': lat_diff,
                'lon_diff': lon_diff,
                'haversine_dist': haversine_dist,
                'manhattan_dist': manhattan_dist,
                'bearing': bearing,
                'jfk_p': jfk_p,
                'jfk_d': jfk_d,
                'lga_p': lga_p,
                'lga_d': lga_d,
                'ewr_p': ewr_p,
                'ewr_d': ewr_d,
                'mid_p': mid_p,
                'mid_d': mid_d,
            },
            index=X.index,
        )
        return features


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    features = X.skb.apply(TaxiFeatureExtractor()).skb.set_name('geo_temporal_features')

    model = skrub.choose_from(
        {
            'lgbm_leaves31': LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=8,
            ),
            'lgbm_leaves63': LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=8,
            ),
        },
        name='lgbm_capacity',
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}