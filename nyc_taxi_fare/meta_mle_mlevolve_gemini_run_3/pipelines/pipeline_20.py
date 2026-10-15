import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

class ClippedRidgeRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=10.0, min_fare=2.5, random_state=42):
        self.alpha = alpha
        self.min_fare = min_fare
        self.random_state = random_state

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        x_scaled = self.scaler_.fit_transform(X)
        self.ridge_ = Ridge(alpha=self.alpha, random_state=self.random_state)
        self.ridge_.fit(x_scaled, y)
        return self

    def predict(self, X):
        x_scaled = self.scaler_.transform(X)
        preds = self.ridge_.predict(x_scaled)
        return np.clip(preds, self.min_fare, None)

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']
    row_keys = setup.get('row_keys')

    dx = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.14
    dy = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.0
    abs_dx = dx.abs()
    abs_dy = dy.abs()
    euclidean_dist = (dx ** 2 + dy ** 2) ** 0.5
    rot_x = 0.8746 * dx - 0.4848 * dy
    rot_y = 0.4848 * dx + 0.8746 * dy
    rot_manhattan = rot_x.abs() + rot_y.abs()

    jfk_p = (((X['pickup_longitude'] - -73.7781) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.6413) * 111.0) ** 2) ** 0.5
    jfk_d = (((X['dropoff_longitude'] - -73.7781) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2) ** 0.5
    is_jfk = ((jfk_p < 2.5).fillna(False) | (jfk_d < 2.5).fillna(False)).astype('float64')

    lga_p = (((X['pickup_longitude'] - -73.874) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.7769) * 111.0) ** 2) ** 0.5
    lga_d = (((X['dropoff_longitude'] - -73.874) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2) ** 0.5
    is_lga = ((lga_p < 2.0).fillna(False) | (lga_d < 2.0).fillna(False)).astype('float64')

    ewr_p = (((X['pickup_longitude'] - -74.1745) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.6895) * 111.0) ** 2) ** 0.5
    ewr_d = (((X['dropoff_longitude'] - -74.1745) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2) ** 0.5
    is_ewr = ((ewr_p < 3.0).fillna(False) | (ewr_d < 3.0).fillna(False)).astype('float64')

    spatial_features = X[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].assign(
        abs_dx=abs_dx,
        abs_dy=abs_dy,
        euclidean_dist=euclidean_dist,
        rot_manhattan=rot_manhattan,
        jfk_min_dist=jfk_p,
        lga_min_dist=lga_p,
        ewr_min_dist=ewr_p,
        is_jfk=is_jfk,
        is_lga=is_lga,
        is_ewr=is_ewr
    ).skb.set_name('spatial_features')

    pickup_dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    hour = pickup_dt.dt.hour.astype('float64')
    dayofweek = pickup_dt.dt.dayofweek.astype('float64')
    year = pickup_dt.dt.year.astype('float64')
    passenger_count = X['passenger_count'].astype('float64')

    is_weekday = (dayofweek <= 4.0).fillna(False)
    is_weekend = (dayofweek >= 5.0).fillna(False).astype('float64')
    is_rush_hour = (is_weekday & (hour >= 16.0).fillna(False) & (hour < 20.0).fillna(False)).astype('float64')
    is_overnight = ((hour >= 20.0).fillna(False) | (hour < 6.0).fillna(False)).astype('float64')

    hour_sin = (2.0 * np.pi * hour / 24.0).skb.apply_func(np.sin)
    hour_cos = (2.0 * np.pi * hour / 24.0).skb.apply_func(np.cos)
    dow_sin = (2.0 * np.pi * dayofweek / 7.0).skb.apply_func(np.sin)
    dow_cos = (2.0 * np.pi * dayofweek / 7.0).skb.apply_func(np.cos)

    temporal_features = spatial_features.assign(
        year=year,
        passenger_count=passenger_count,
        is_rush_hour=is_rush_hour,
        is_overnight=is_overnight,
        is_weekend=is_weekend,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        dow_sin=dow_sin,
        dow_cos=dow_cos
    ).skb.set_name('temporal_features')

    is_post_2012_hike = (X['pickup_datetime'] >= '2012-09-04').fillna(False).astype('float64')
    dist_x_post2012 = rot_manhattan * is_post_2012_hike

    est_meter_fare = 2.5 + (1.24275 + 0.31068 * is_post_2012_hike) * rot_manhattan + 1.0 * is_rush_hour + 0.5 * is_overnight
    est_jfk_flat_fare = is_jfk * (45.0 + 7.0 * is_post_2012_hike + 5.33)
    est_ewr_fare = is_ewr * (est_meter_fare + 15.0 + 2.5 * is_post_2012_hike + 11.0)
    sqrt_dist = rot_manhattan ** 0.5
    log_dist = rot_manhattan.skb.apply_func(np.log1p)

    tariff_features = temporal_features.assign(
        is_post_2012_hike=is_post_2012_hike,
        dist_x_post2012=dist_x_post2012,
        est_meter_fare=est_meter_fare,
        est_jfk_flat_fare=est_jfk_flat_fare,
        est_ewr_fare=est_ewr_fare,
        sqrt_dist=sqrt_dist,
        log_dist=log_dist
    ).skb.set_name('tariff_features')

    model = ClippedRidgeRegressor(alpha=10.0, min_fare=2.5, random_state=42)
    pred = tariff_features.skb.apply(model, y=y).skb.set_name('model')

    result = {'pred': pred, 'scoring': scoring}
    if row_keys is not None:
        result['row_keys'] = row_keys
    return result