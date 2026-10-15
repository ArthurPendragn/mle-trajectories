import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

class ClippedRidgeRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1.0, min_fare=2.5, random_state=42):
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
    abs_rot_x = rot_x.abs()
    abs_rot_y = rot_y.abs()
    rot_manhattan = abs_rot_x + abs_rot_y

    p_dx_mid = (X['pickup_longitude'] - -73.9855) * 84.14
    p_dy_mid = (X['pickup_latitude'] - 40.758) * 111.0
    p_rot_x = 0.8746 * p_dx_mid - 0.4848 * p_dy_mid
    p_rot_y = 0.4848 * p_dx_mid + 0.8746 * p_dy_mid
    d_dx_mid = (X['dropoff_longitude'] - -73.9855) * 84.14
    d_dy_mid = (X['dropoff_latitude'] - 40.758) * 111.0
    d_rot_x = 0.8746 * d_dx_mid - 0.4848 * d_dy_mid
    d_rot_y = 0.4848 * d_dx_mid + 0.8746 * d_dy_mid

    in_manhattan_p = ((p_rot_x >= -3.5) & (p_rot_x <= 3.5) & (p_rot_y >= -12.0) & (p_rot_y <= 12.0)).fillna(False).astype('float64')
    in_manhattan_d = ((d_rot_x >= -3.5) & (d_rot_x <= 3.5) & (d_rot_y >= -12.0) & (d_rot_y <= 12.0)).fillna(False).astype('float64')
    is_intra_manhattan = ((in_manhattan_p == 1.0) & (in_manhattan_d == 1.0)).fillna(False).astype('float64')
    is_borough_crossing = (in_manhattan_p - in_manhattan_d).abs()

    jfk_p = (((X['pickup_longitude'] - -73.7781) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.6413) * 111.0) ** 2) ** 0.5
    jfk_d = (((X['dropoff_longitude'] - -73.7781) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2) ** 0.5
    is_jfk = ((jfk_p < 2.5).fillna(False) | (jfk_d < 2.5).fillna(False)).astype('float64')

    lga_p = (((X['pickup_longitude'] - -73.874) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.7769) * 111.0) ** 2) ** 0.5
    lga_d = (((X['dropoff_longitude'] - -73.874) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2) ** 0.5
    is_lga = ((lga_p < 2.0).fillna(False) | (lga_d < 2.0).fillna(False)).astype('float64')

    ewr_p = (((X['pickup_longitude'] - -74.1745) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.6895) * 111.0) ** 2) ** 0.5
    ewr_d = (((X['dropoff_longitude'] - -74.1745) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2) ** 0.5
    is_ewr = ((ewr_p < 3.0).fillna(False) | (ewr_d < 3.0).fillna(False)).astype('float64')

    lincoln_p = (((X['pickup_longitude'] - -74.0044) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.7615) * 111.0) ** 2) ** 0.5
    lincoln_d = (((X['dropoff_longitude'] - -74.0044) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.7615) * 111.0) ** 2) ** 0.5
    lincoln_detour = lincoln_p + lincoln_d - euclidean_dist

    holland_p = (((X['pickup_longitude'] - -74.0216) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.7277) * 111.0) ** 2) ** 0.5
    holland_d = (((X['dropoff_longitude'] - -74.0216) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.7277) * 111.0) ** 2) ** 0.5
    holland_detour = holland_p + holland_d - euclidean_dist

    qmt_p = (((X['pickup_longitude'] - -73.9645) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.7445) * 111.0) ** 2) ** 0.5
    qmt_d = (((X['dropoff_longitude'] - -73.9645) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.7445) * 111.0) ** 2) ** 0.5
    qmt_detour = qmt_p + qmt_d - euclidean_dist

    rfk_p = (((X['pickup_longitude'] - -73.9282) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.7798) * 111.0) ** 2) ** 0.5
    rfk_d = (((X['dropoff_longitude'] - -73.9282) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.7798) * 111.0) ** 2) ** 0.5
    rfk_detour = rfk_p + rfk_d - euclidean_dist

    is_hudson_tolled = ((lincoln_detour < 1.5).fillna(False) | (holland_detour < 1.5).fillna(False)).astype('float64')
    is_east_river_tolled = ((qmt_detour < 1.5).fillna(False) | (rfk_detour < 1.5).fillna(False)).astype('float64')

    spatial_features = X[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].assign(
        abs_dx=abs_dx,
        abs_dy=abs_dy,
        euclidean_dist=euclidean_dist,
        rot_manhattan=rot_manhattan,
        abs_rot_x=abs_rot_x,
        abs_rot_y=abs_rot_y,
        in_manhattan_p=in_manhattan_p,
        in_manhattan_d=in_manhattan_d,
        is_intra_manhattan=is_intra_manhattan,
        is_borough_crossing=is_borough_crossing,
        jfk_min_dist=jfk_p,
        lga_min_dist=lga_p,
        ewr_min_dist=ewr_p,
        is_jfk=is_jfk,
        is_lga=is_lga,
        is_ewr=is_ewr,
        lincoln_detour=lincoln_detour,
        holland_detour=holland_detour,
        qmt_detour=qmt_detour,
        rfk_detour=rfk_detour,
        is_hudson_tolled=is_hudson_tolled,
        is_east_river_tolled=is_east_river_tolled
    ).skb.set_name('spatial_features')

    pickup_dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    hour = pickup_dt.dt.hour.astype('float64')
    dayofweek = pickup_dt.dt.dayofweek.astype('float64')
    year = pickup_dt.dt.year.astype('float64')
    passenger_count = X['passenger_count'].astype('float64')

    is_weekday = (dayofweek <= 4.0).fillna(False)
    is_weekend = (dayofweek >= 5.0).fillna(False).astype('float64')
    is_morning_rush = (is_weekday & (hour >= 7.0).fillna(False) & (hour < 10.0).fillna(False)).astype('float64')
    is_rush_hour = (is_weekday & (hour >= 16.0).fillna(False) & (hour < 20.0).fillna(False)).astype('float64')
    is_overnight = ((hour >= 20.0).fillna(False) | (hour < 6.0).fillna(False)).astype('float64')

    hour_sin = (2.0 * np.pi * hour / 24.0).skb.apply_func(np.sin)
    hour_cos = (2.0 * np.pi * hour / 24.0).skb.apply_func(np.cos)
    dow_sin = (2.0 * np.pi * dayofweek / 7.0).skb.apply_func(np.sin)
    dow_cos = (2.0 * np.pi * dayofweek / 7.0).skb.apply_func(np.cos)

    temporal_features = spatial_features.assign(
        year=year,
        passenger_count=passenger_count,
        is_morning_rush=is_morning_rush,
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
    dist_x_intra_manhattan = rot_manhattan * is_intra_manhattan
    dist_x_intra_manhattan_post2012 = dist_x_intra_manhattan * is_post_2012_hike
    dist_x_rush_hour = rot_manhattan * is_rush_hour
    dist_x_morning_rush = rot_manhattan * is_morning_rush

    abs_rot_x_intra = abs_rot_x * is_intra_manhattan
    abs_rot_y_intra = abs_rot_y * is_intra_manhattan

    diff3 = 3.0 - rot_manhattan
    diff3_pos = (diff3 + diff3.abs()) * 0.5
    dist_short = 3.0 - diff3_pos

    diff_m = rot_manhattan - 3.0
    diff_m_pos = (diff_m + diff_m.abs()) * 0.5
    diff_m_hi = rot_manhattan - 10.0
    diff_m_hi_pos = (diff_m_hi + diff_m_hi.abs()) * 0.5
    dist_medium = diff_m_pos - diff_m_hi_pos
    dist_long = diff_m_hi_pos

    is_jfk_flat = (is_jfk * ((in_manhattan_p == 1.0) | (in_manhattan_d == 1.0)).fillna(False)).astype('float64')
    est_meter_fare = 2.5 + (1.24275 + 0.31068 * is_post_2012_hike) * rot_manhattan + 1.0 * is_rush_hour + 0.5 * is_overnight
    est_jfk_flat_fare = is_jfk_flat * (45.0 + 7.0 * is_post_2012_hike + 5.33)
    est_ewr_fare = is_ewr * (est_meter_fare + 15.0 + 2.5 * is_post_2012_hike + 11.0)
    sqrt_dist = rot_manhattan ** 0.5
    log_dist = rot_manhattan.skb.apply_func(np.log1p)

    tariff_features = temporal_features.assign(
        is_post_2012_hike=is_post_2012_hike,
        dist_x_post2012=dist_x_post2012,
        dist_x_intra_manhattan=dist_x_intra_manhattan,
        dist_x_intra_manhattan_post2012=dist_x_intra_manhattan_post2012,
        dist_x_rush_hour=dist_x_rush_hour,
        dist_x_morning_rush=dist_x_morning_rush,
        abs_rot_x_intra=abs_rot_x_intra,
        abs_rot_y_intra=abs_rot_y_intra,
        dist_short=dist_short,
        dist_medium=dist_medium,
        dist_long=dist_long,
        is_jfk_flat=is_jfk_flat,
        est_meter_fare=est_meter_fare,
        est_jfk_flat_fare=est_jfk_flat_fare,
        est_ewr_fare=est_ewr_fare,
        sqrt_dist=sqrt_dist,
        log_dist=log_dist
    ).skb.set_name('tariff_features')

    alpha = skrub.choose_from({'alpha_1': 1.0, 'alpha_5': 5.0, 'alpha_10': 10.0}, name='ridge_alpha')
    model = ClippedRidgeRegressor(alpha=alpha, min_fare=2.5, random_state=42)
    pred = tariff_features.skb.apply(model, y=y).skb.set_name('model')

    result = {'pred': pred, 'scoring': scoring}
    if row_keys is not None:
        result['row_keys'] = row_keys
    return result