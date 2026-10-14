import lightgbm as lgb
import numpy as np
import skrub
import xgboost as xgb
from sklearn.ensemble import VotingRegressor


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    year = X['pickup_datetime'].str.slice(0, 4).astype('int32')
    month = X['pickup_datetime'].str.slice(5, 7).astype('int32')
    day = X['pickup_datetime'].str.slice(8, 10).astype('int32')
    hour = X['pickup_datetime'].str.slice(11, 13).astype('int32')

    month_offset = month.map({
        1: 0,
        2: 3,
        3: 2,
        4: 5,
        5: 0,
        6: 3,
        7: 5,
        8: 1,
        9: 4,
        10: 6,
        11: 2,
        12: 4,
    })
    y_adj = year - (month < 3).fillna(False).astype('int32')
    day_of_week = (
        y_adj
        + y_adj // 4
        - y_adj // 100
        + y_adj // 400
        + month_offset
        + day
    ) % 7
    is_weekend = (
        ((day_of_week == 0) | (day_of_week == 6)).fillna(False).astype('int32')
    )

    dlon = X['dropoff_longitude'] - X['pickup_longitude']
    dlat = X['dropoff_latitude'] - X['pickup_latitude']
    bearing = dlat.skb.apply_func(np.arctan2, dlon * 0.75756)
    abs_dlon = dlon.abs()
    abs_dlat = dlat.abs()
    euclidean_dist = ((abs_dlon * 0.75756) ** 2 + abs_dlat**2) ** 0.5 * 111.0
    manhattan_dist = abs_dlon * 84.09 + abs_dlat * 111.0

    cos_29 = 0.8746197
    sin_29 = 0.4848096
    dx = dlon * 0.75756
    dy = dlat
    dx_rot = dx * cos_29 - dy * sin_29
    dy_rot = dx * sin_29 + dy * cos_29
    rot_manhattan_km = (dx_rot.abs() + dy_rot.abs()) * 111.0

    rot_bearing = dy_rot.skb.apply_func(np.arctan2, dx_rot)
    circuity = rot_manhattan_km / (euclidean_dist + 0.001)

    px_norm = (X['pickup_longitude'] - -74.0) * 0.75756
    py_norm = X['pickup_latitude'] - 40.7
    p_x_rot = px_norm * cos_29 - py_norm * sin_29
    p_y_rot = px_norm * sin_29 + py_norm * cos_29

    dx_norm = (X['dropoff_longitude'] - -74.0) * 0.75756
    dy_norm = X['dropoff_latitude'] - 40.7
    d_x_rot = dx_norm * cos_29 - dy_norm * sin_29
    d_y_rot = dx_norm * sin_29 + dy_norm * cos_29

    p_jfk = (
        ((X['pickup_longitude'] - -73.7781) * 0.75756) ** 2
        + (X['pickup_latitude'] - 40.6413) ** 2
    ) ** 0.5 * 111.0
    d_jfk = (
        ((X['dropoff_longitude'] - -73.7781) * 0.75756) ** 2
        + (X['dropoff_latitude'] - 40.6413) ** 2
    ) ** 0.5 * 111.0
    p_lga = (
        ((X['pickup_longitude'] - -73.8740) * 0.75756) ** 2
        + (X['pickup_latitude'] - 40.7769) ** 2
    ) ** 0.5 * 111.0
    d_lga = (
        ((X['dropoff_longitude'] - -73.8740) * 0.75756) ** 2
        + (X['dropoff_latitude'] - 40.7769) ** 2
    ) ** 0.5 * 111.0
    p_ewr = (
        ((X['pickup_longitude'] - -74.1745) * 0.75756) ** 2
        + (X['pickup_latitude'] - 40.6895) ** 2
    ) ** 0.5 * 111.0
    d_ewr = (
        ((X['dropoff_longitude'] - -74.1745) * 0.75756) ** 2
        + (X['dropoff_latitude'] - 40.6895) ** 2
    ) ** 0.5 * 111.0
    p_mid = (
        ((X['pickup_longitude'] - -73.9855) * 0.75756) ** 2
        + (X['pickup_latitude'] - 40.7580) ** 2
    ) ** 0.5 * 111.0
    d_mid = (
        ((X['dropoff_longitude'] - -73.9855) * 0.75756) ** 2
        + (X['dropoff_latitude'] - 40.7580) ** 2
    ) ** 0.5 * 111.0

    is_overnight = ((hour >= 20) | (hour < 6)).fillna(False).astype('int32')
    is_rush_hour = (
        ((hour >= 16) & (hour < 20) & (is_weekend == 0))
        .fillna(False)
        .astype('int32')
    )
    is_jfk = ((p_jfk < 3.0) | (d_jfk < 3.0)).fillna(False).astype('int32')
    is_lga = ((p_lga < 3.0) | (d_lga < 3.0)).fillna(False).astype('int32')
    is_ewr = ((p_ewr < 3.0) | (d_ewr < 3.0)).fillna(False).astype('int32')

    is_jfk_flat = (
        ((p_jfk < 3.0) & (d_mid < 3.0)) | ((d_jfk < 3.0) & (p_mid < 3.0))
    ).fillna(False).astype('int32')

    date_int = year * 10000 + month * 100 + day
    is_post_2012 = (date_int >= 20120904).fillna(False).astype('int32')

    all_features = X[[
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count',
    ]].assign(
        year=year,
        month=month,
        day=day,
        hour=hour,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
        dlon=dlon,
        dlat=dlat,
        bearing=bearing,
        abs_dlon=abs_dlon,
        abs_dlat=abs_dlat,
        euclidean_km=euclidean_dist,
        manhattan_km=manhattan_dist,
        dx_rot=dx_rot,
        dy_rot=dy_rot,
        rot_manhattan_km=rot_manhattan_km,
        p_jfk_km=p_jfk,
        d_jfk_km=d_jfk,
        p_lga_km=p_lga,
        d_lga_km=d_lga,
        p_ewr_km=p_ewr,
        d_ewr_km=d_ewr,
        p_mid_km=p_mid,
        d_mid_km=d_mid,
        is_overnight=is_overnight,
        is_rush_hour=is_rush_hour,
        is_jfk=is_jfk,
        is_lga=is_lga,
        is_ewr=is_ewr,
        p_x_rot=p_x_rot,
        p_y_rot=p_y_rot,
        d_x_rot=d_x_rot,
        d_y_rot=d_y_rot,
        rot_bearing=rot_bearing,
        circuity=circuity,
        is_jfk_flat=is_jfk_flat,
        is_post_2012=is_post_2012,
    )

    parent_cols = [
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count',
        'year',
        'month',
        'day',
        'hour',
        'day_of_week',
        'is_weekend',
        'dlon',
        'dlat',
        'bearing',
        'abs_dlon',
        'abs_dlat',
        'euclidean_km',
        'manhattan_km',
        'p_jfk_km',
        'd_jfk_km',
        'p_lga_km',
        'd_lga_km',
        'p_ewr_km',
        'd_ewr_km',
        'p_mid_km',
        'd_mid_km',
    ]
    rotated_delta_and_tlc_cols = parent_cols + [
        'dx_rot',
        'dy_rot',
        'rot_manhattan_km',
        'is_overnight',
        'is_rush_hour',
        'is_jfk',
        'is_lga',
        'is_ewr',
    ]
    rotated_coords_and_tlc_cols = rotated_delta_and_tlc_cols + [
        'p_x_rot',
        'p_y_rot',
        'd_x_rot',
        'd_y_rot',
    ]

    rot_spatial_enhancements_cols = rotated_coords_and_tlc_cols + [
        'rot_bearing',
        'circuity',
        'is_jfk_flat',
    ]
    rot_spatial_and_fare_hike_cols = rot_spatial_enhancements_cols + [
        'is_post_2012'
    ]

    feature_choices = {
        'rot_spatial_enhancements': rot_spatial_enhancements_cols,
        'rot_spatial_and_fare_hike': rot_spatial_and_fare_hike_cols,
    }
    col_choice = skrub.choose_from(feature_choices, name='feature_variant')

    features = all_features[col_choice].skb.set_name('features')

    lgb_est = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=95,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )
    xgb_est_10 = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda',
        random_state=42,
    )
    model = VotingRegressor(
        estimators=[('lgb', lgb_est), ('xgb10', xgb_est_10)], weights=[0.4, 0.6]
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}