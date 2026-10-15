import pandas as pd
import skrub
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']
    row_keys = setup.get('row_keys')

    # 1. Spatial features: physical grid displacements, rotated Manhattan distance, and airport proximity
    dx = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.14
    dy = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.0
    abs_dx = dx.abs()
    abs_dy = dy.abs()
    euclidean_dist = ((dx ** 2) + (dy ** 2)) ** 0.5

    rot_x = 0.8746 * dx - 0.4848 * dy
    rot_y = 0.4848 * dx + 0.8746 * dy
    rot_manhattan = rot_x.abs() + rot_y.abs()

    jfk_p = (((X['pickup_longitude'] - (-73.7781)) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.6413) * 111.0) ** 2) ** 0.5
    jfk_d = (((X['dropoff_longitude'] - (-73.7781)) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2) ** 0.5
    is_jfk = ((jfk_p < 2.5).fillna(False) | (jfk_d < 2.5).fillna(False)).astype('float64')

    lga_p = (((X['pickup_longitude'] - (-73.8740)) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.7769) * 111.0) ** 2) ** 0.5
    lga_d = (((X['dropoff_longitude'] - (-73.8740)) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2) ** 0.5
    is_lga = ((lga_p < 2.0).fillna(False) | (lga_d < 2.0).fillna(False)).astype('float64')

    ewr_p = (((X['pickup_longitude'] - (-74.1745)) * 84.14) ** 2 + ((X['pickup_latitude'] - 40.6895) * 111.0) ** 2) ** 0.5
    ewr_d = (((X['dropoff_longitude'] - (-74.1745)) * 84.14) ** 2 + ((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2) ** 0.5
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
        is_ewr=is_ewr,
    ).skb.set_name('spatial_features')

    # 2. Temporal features: calendar cycles, peak rush hour and overnight surcharge flags
    pickup_dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    hour = pickup_dt.dt.hour.astype('float64')
    dayofweek = pickup_dt.dt.dayofweek.astype('float64')
    year = pickup_dt.dt.year.astype('float64')
    passenger_count = X['passenger_count'].astype('float64')

    is_weekday = (dayofweek <= 4.0).fillna(False)
    is_rush_hour = (is_weekday & (hour >= 16.0).fillna(False) & (hour < 20.0).fillna(False)).astype('float64')
    is_overnight = ((hour >= 20.0).fillna(False) | (hour < 6.0).fillna(False)).astype('float64')

    temporal_features = spatial_features.assign(
        hour=hour,
        dayofweek=dayofweek,
        year=year,
        passenger_count=passenger_count,
        is_rush_hour=is_rush_hour,
        is_overnight=is_overnight,
    ).skb.set_name('temporal_features')

    # 3. Tariff interaction features: statutory 2012 rate hike and additive surcharge distance interactions
    is_post_2012_hike = (X['pickup_datetime'] >= '2012-09-04').fillna(False).astype('float64')
    dist_x_post2012 = rot_manhattan * is_post_2012_hike
    dist_x_rush = rot_manhattan * is_rush_hour
    dist_x_overnight = rot_manhattan * is_overnight

    tariff_features = temporal_features.assign(
        is_post_2012_hike=is_post_2012_hike,
        dist_x_post2012=dist_x_post2012,
        dist_x_rush=dist_x_rush,
        dist_x_overnight=dist_x_overnight,
    ).skb.set_name('tariff_features')

    # 4. Standardized regularized linear model (Ridge regression)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0, random_state=42)),
    ])
    pred = tariff_features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': scoring,
        'row_keys': row_keys,
    }