import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, format='%Y-%m-%d %H:%M:%S UTC')
    year = dt.dt.year.astype('int32')
    month = dt.dt.month.astype('int32')
    day = dt.dt.day.astype('int32')
    hour = dt.dt.hour.astype('int32')
    weekday = dt.dt.dayofweek.astype('int32')

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

    bearing = dy_km.skb.apply_func(np.arctan2, dx_km)

    dist_pickup_jfk = (((X['pickup_latitude'] - 40.6413) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.7781) * 84.0) ** 2) ** 0.5
    dist_dropoff_jfk = (((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.7781) * 84.0) ** 2) ** 0.5
    dist_pickup_lga = (((X['pickup_latitude'] - 40.7769) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.8740) * 84.0) ** 2) ** 0.5
    dist_dropoff_lga = (((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.8740) * 84.0) ** 2) ** 0.5
    dist_pickup_ewr = (((X['pickup_latitude'] - 40.6895) * 111.0) ** 2 + ((X['pickup_longitude'] - -74.1745) * 84.0) ** 2) ** 0.5
    dist_dropoff_ewr = (((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2 + ((X['dropoff_longitude'] - -74.1745) * 84.0) ** 2) ** 0.5
    dist_pickup_midtown = (((X['pickup_latitude'] - 40.7580) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.9855) * 84.0) ** 2) ** 0.5
    dist_dropoff_midtown = (((X['dropoff_latitude'] - 40.7580) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.9855) * 84.0) ** 2) ** 0.5

    fare_hike_2012 = ((year > 2012) | ((year == 2012) & (month >= 9))).astype('int32')
    rush_hour_surcharge = ((weekday < 5) & (hour >= 16) & (hour < 20)).astype('int32')
    overnight_surcharge = ((hour >= 20) | (hour < 6)).astype('int32')

    hour_rad = hour * (2.0 * np.pi / 24.0)
    hour_sin = hour_rad.skb.apply_func(np.sin)
    hour_cos = hour_rad.skb.apply_func(np.cos)

    jfk_trip = ((dist_pickup_jfk < 3.0) | (dist_dropoff_jfk < 3.0)).astype('int32')
    lga_trip = ((dist_pickup_lga < 3.0) | (dist_dropoff_lga < 3.0)).astype('int32')
    ewr_trip = ((dist_pickup_ewr < 3.0) | (dist_dropoff_ewr < 3.0)).astype('int32')
    jfk_manhattan_flat = (
        ((dist_pickup_jfk < 3.0) & (dist_dropoff_midtown < 5.0))
        | ((dist_dropoff_jfk < 3.0) & (dist_pickup_midtown < 5.0))
    ).astype('int32')

    dist_x_fare_hike = rotated_manhattan_dist * fare_hike_2012
    is_stationary = (haversine_dist < 0.05).astype('int32')

    features = X.drop(columns=['key', 'pickup_datetime']).assign(
        abs_diff_lon=d_lon,
        abs_diff_lat=d_lat,
        euclidean_dist=euclidean_dist,
        haversine_dist=haversine_dist,
        rotated_manhattan_dist=rotated_manhattan_dist,
        bearing=bearing,
        dist_pickup_jfk=dist_pickup_jfk,
        dist_dropoff_jfk=dist_dropoff_jfk,
        dist_pickup_lga=dist_pickup_lga,
        dist_dropoff_lga=dist_dropoff_lga,
        dist_pickup_ewr=dist_pickup_ewr,
        dist_dropoff_ewr=dist_dropoff_ewr,
        dist_pickup_midtown=dist_pickup_midtown,
        dist_dropoff_midtown=dist_dropoff_midtown,
        year=year,
        month=month,
        day=day,
        hour=hour,
        weekday=weekday,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        fare_hike_2012=fare_hike_2012,
        rush_hour_surcharge=rush_hour_surcharge,
        overnight_surcharge=overnight_surcharge,
        jfk_trip=jfk_trip,
        lga_trip=lga_trip,
        ewr_trip=ewr_trip,
        jfk_manhattan_flat=jfk_manhattan_flat,
        dist_x_fare_hike=dist_x_fare_hike,
        is_stationary=is_stationary,
    ).skb.set_name('features')

    lgb_est = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    ridge_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0)),
    ])
    stack_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    model = StackingRegressor(
        estimators=[('lgb', lgb_est), ('ridge', ridge_pipe)],
        final_estimator=Ridge(alpha=1.0),
        cv=stack_cv,
        passthrough=False,
    )
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}