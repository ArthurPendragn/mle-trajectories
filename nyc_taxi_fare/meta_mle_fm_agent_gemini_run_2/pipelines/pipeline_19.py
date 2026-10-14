import lightgbm as lgb
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import VotingRegressor
import xgboost as xgb


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    dt = X['pickup_datetime'].skb.apply_func(
        pd.to_datetime, format='%Y-%m-%d %H:%M:%S UTC'
    )
    year = dt.dt.year.astype('int32')
    month = dt.dt.month.astype('int32')
    day = dt.dt.day.astype('int32')
    hour = dt.dt.hour.astype('int32')
    weekday = dt.dt.dayofweek.astype('int32')

    d_lon = (X['dropoff_longitude'] - X['pickup_longitude']).abs()
    d_lat = (X['dropoff_latitude'] - X['pickup_latitude']).abs()
    dx_km = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.0
    dy_km = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.0
    euclidean_dist = (dy_km**2 + dx_km**2) ** 0.5

    p_lat_rad = X['pickup_latitude'] * (np.pi / 180.0)
    d_lat_rad = X['dropoff_latitude'] * (np.pi / 180.0)
    d_lon_rad = (X['dropoff_longitude'] - X['pickup_longitude']) * (
        np.pi / 180.0
    )
    d_lat_diff = (X['dropoff_latitude'] - X['pickup_latitude']) * (
        np.pi / 180.0
    )
    sin_dlat = (d_lat_diff * 0.5).skb.apply_func(np.sin)
    sin_dlon = (d_lon_rad * 0.5).skb.apply_func(np.sin)
    cos_plat = p_lat_rad.skb.apply_func(np.cos)
    cos_dlat = d_lat_rad.skb.apply_func(np.cos)
    a = sin_dlat * sin_dlat + cos_plat * cos_dlat * sin_dlon * sin_dlon
    a_clipped = a.skb.apply_func(np.clip, 0.0, 1.0)
    haversine_dist = (
        a_clipped.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin)
        * (2.0 * 6371.0)
    )

    theta = 29.0 * (np.pi / 180.0)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    dx_rot = dx_km * cos_theta - dy_km * sin_theta
    dy_rot = dx_km * sin_theta + dy_km * cos_theta
    rotated_manhattan_dist = dx_rot.abs() + dy_rot.abs()
    bearing = dy_km.skb.apply_func(np.arctan2, dx_km)

    dist_pickup_jfk = (
        ((X['pickup_latitude'] - 40.6413) * 111.0) ** 2
        + ((X['pickup_longitude'] - -73.7781) * 84.0) ** 2
    ) ** 0.5
    dist_dropoff_jfk = (
        ((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2
        + ((X['dropoff_longitude'] - -73.7781) * 84.0) ** 2
    ) ** 0.5
    dist_pickup_lga = (
        ((X['pickup_latitude'] - 40.7769) * 111.0) ** 2
        + ((X['pickup_longitude'] - -73.874) * 84.0) ** 2
    ) ** 0.5
    dist_dropoff_lga = (
        ((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2
        + ((X['dropoff_longitude'] - -73.874) * 84.0) ** 2
    ) ** 0.5
    dist_pickup_ewr = (
        ((X['pickup_latitude'] - 40.6895) * 111.0) ** 2
        + ((X['pickup_longitude'] - -74.1745) * 84.0) ** 2
    ) ** 0.5
    dist_dropoff_ewr = (
        ((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2
        + ((X['dropoff_longitude'] - -74.1745) * 84.0) ** 2
    ) ** 0.5
    dist_pickup_midtown = (
        ((X['pickup_latitude'] - 40.758) * 111.0) ** 2
        + ((X['pickup_longitude'] - -73.9855) * 84.0) ** 2
    ) ** 0.5
    dist_dropoff_midtown = (
        ((X['dropoff_latitude'] - 40.758) * 111.0) ** 2
        + ((X['dropoff_longitude'] - -73.9855) * 84.0) ** 2
    ) ** 0.5

    fare_hike_2012 = ((year > 2012) | (year == 2012) & (month >= 9)).astype(
        'int32'
    )
    rush_hour_surcharge = (
        (weekday < 5) & (hour >= 16) & (hour < 20)
    ).astype('int32')
    overnight_surcharge = ((hour >= 20) | (hour < 6)).astype('int32')

    features = (
        X.drop(columns=['key', 'pickup_datetime'])
        .assign(
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
            fare_hike_2012=fare_hike_2012,
            rush_hour_surcharge=rush_hour_surcharge,
            overnight_surcharge=overnight_surcharge,
        )
        .skb.set_name('features')
    )

    est_600_lgb = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    est_600_xgb = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        device='cuda',
        n_jobs=-1,
    )

    est_800_lgb = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    est_800_xgb = xgb.XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist',
        device='cuda',
        n_jobs=-1,
    )

    model_choice = skrub.choose_from(
        {
            'capacity_600_lr04': VotingRegressor(
                estimators=[('lgb', est_600_lgb), ('xgb', est_600_xgb)],
                weights=[0.65, 0.35],
            ),
            'capacity_800_lr03': VotingRegressor(
                estimators=[('lgb', est_800_lgb), ('xgb', est_800_xgb)],
                weights=[0.65, 0.35],
            ),
        },
        name='capacity_config',
    )

    pred = features.skb.apply(model_choice, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}