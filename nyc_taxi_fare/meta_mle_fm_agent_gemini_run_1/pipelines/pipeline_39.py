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
    minute = X['pickup_datetime'].str.slice(14, 16).astype('int32')
    pickup_hour_float = hour.astype('float64') + minute.astype('float64') / 60.0

    month_offset = month.map({
        1: 0, 2: 3, 3: 2, 4: 5, 5: 0, 6: 3,
        7: 5, 8: 1, 9: 4, 10: 6, 11: 2, 12: 4,
    })
    y_adj = year - (month < 3).fillna(False).astype('int32')
    day_of_week = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + month_offset + day) % 7
    is_weekend = ((day_of_week == 0) | (day_of_week == 6)).fillna(False).astype('int32')

    dlon = X['dropoff_longitude'] - X['pickup_longitude']
    dlat = X['dropoff_latitude'] - X['pickup_latitude']
    bearing = dlat.skb.apply_func(np.arctan2, dlon * 0.75756)
    abs_dlon = dlon.abs()
    abs_dlat = dlat.abs()
    euclidean_dist = ((abs_dlon * 0.75756) ** 2 + abs_dlat ** 2) ** 0.5 * 111.0
    manhattan_dist = abs_dlon * 84.09 + abs_dlat * 111.0

    cos_29 = 0.8746197
    sin_29 = 0.4848096
    dx = dlon * 0.75756
    dy = dlat
    dx_rot = dx * cos_29 - dy * sin_29
    dy_rot = dx * sin_29 + dy * cos_29
    rot_manhattan_km = (dx_rot.abs() + dy_rot.abs()) * 111.0

    px_norm = (X['pickup_longitude'] - -74.0) * 0.75756
    py_norm = X['pickup_latitude'] - 40.7
    p_x_rot = px_norm * cos_29 - py_norm * sin_29
    p_y_rot = px_norm * sin_29 + py_norm * cos_29

    dx_norm = (X['dropoff_longitude'] - -74.0) * 0.75756
    dy_norm = X['dropoff_latitude'] - 40.7
    d_x_rot = dx_norm * cos_29 - dy_norm * sin_29
    d_y_rot = dx_norm * sin_29 + dy_norm * cos_29

    p_jfk = (((X['pickup_longitude'] - -73.7781) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.6413) ** 2) ** 0.5 * 111.0
    d_jfk = (((X['dropoff_longitude'] - -73.7781) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.6413) ** 2) ** 0.5 * 111.0
    p_lga = (((X['pickup_longitude'] - -73.8740) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.7769) ** 2) ** 0.5 * 111.0
    d_lga = (((X['dropoff_longitude'] - -73.8740) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.7769) ** 2) ** 0.5 * 111.0
    p_ewr = (((X['pickup_longitude'] - -74.1745) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.6895) ** 2) ** 0.5 * 111.0
    d_ewr = (((X['dropoff_longitude'] - -74.1745) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.6895) ** 2) ** 0.5 * 111.0
    p_mid = (((X['pickup_longitude'] - -73.9855) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.7580) ** 2) ** 0.5 * 111.0
    d_mid = (((X['dropoff_longitude'] - -73.9855) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.7580) ** 2) ** 0.5 * 111.0

    is_overnight = ((hour >= 20) | (hour < 6)).fillna(False).astype('int32')
    is_rush_hour = ((hour >= 16) & (hour < 20) & (is_weekend == 0)).fillna(False).astype('int32')
    is_jfk = ((p_jfk < 3.0) | (d_jfk < 3.0)).fillna(False).astype('int32')
    is_lga = ((p_lga < 3.0) | (d_lga < 3.0)).fillna(False).astype('int32')
    is_ewr = ((p_ewr < 3.0) | (d_ewr < 3.0)).fillna(False).astype('int32')

    date_int = year * 10000 + month * 100 + day
    is_post_hike_2012 = (date_int >= 20120904).fillna(False).astype('int32')
    dist_post_hike = rot_manhattan_km * is_post_hike_2012
    is_post_mta_2009 = (date_int >= 20091101).fillna(False).astype('int32')
    is_jfk_flat = (((p_jfk < 3.0) & (d_mid < 9.0)) | ((d_jfk < 3.0) & (p_mid < 9.0))).fillna(False).astype('int32')
    jfk_post_hike = is_jfk_flat * is_post_hike_2012
    elapsed_days = (year - 2009) * 365 + (month - 1) * 30 + day

    is_p_east = (X['pickup_longitude'] > -73.96).fillna(False).astype('int32')
    is_d_east = (X['dropoff_longitude'] > -73.96).fillna(False).astype('int32')
    cross_east_river = (is_p_east != is_d_east).fillna(False).astype('int32')
    circuity = rot_manhattan_km / (euclidean_dist + 0.001)

    dist_rush_hour = rot_manhattan_km * is_rush_hour
    dist_cross_river = rot_manhattan_km * cross_east_river
    is_intra_manhattan = ((p_mid < 8.0) & (d_mid < 8.0) & (cross_east_river == 0)).fillna(False).astype('int32')
    dist_intra_manhattan = rot_manhattan_km * is_intra_manhattan

    all_features = X[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']].assign(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        pickup_hour_float=pickup_hour_float,
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
        is_post_hike_2012=is_post_hike_2012,
        dist_post_hike=dist_post_hike,
        is_post_mta_2009=is_post_mta_2009,
        is_jfk_flat=is_jfk_flat,
        jfk_post_hike=jfk_post_hike,
        elapsed_days=elapsed_days,
        cross_east_river=cross_east_river,
        circuity=circuity,
        dist_rush_hour=dist_rush_hour,
        dist_cross_river=dist_cross_river,
        is_intra_manhattan=is_intra_manhattan,
        dist_intra_manhattan=dist_intra_manhattan,
    )

    base_cols = [
        'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
        'year', 'month', 'day', 'hour', 'pickup_hour_float', 'day_of_week', 'is_weekend',
        'dlon', 'dlat', 'bearing', 'abs_dlon', 'abs_dlat', 'euclidean_km', 'manhattan_km',
        'p_jfk_km', 'd_jfk_km', 'p_lga_km', 'd_lga_km', 'p_ewr_km', 'd_ewr_km', 'p_mid_km', 'd_mid_km',
        'dx_rot', 'dy_rot', 'rot_manhattan_km', 'p_x_rot', 'p_y_rot', 'd_x_rot', 'd_y_rot',
        'is_overnight', 'is_rush_hour', 'is_jfk', 'is_lga', 'is_ewr',
        'is_post_hike_2012', 'dist_post_hike', 'is_post_mta_2009', 'is_jfk_flat', 'jfk_post_hike',
        'elapsed_days', 'cross_east_river', 'circuity',
        'dist_rush_hour', 'dist_cross_river',
    ]

    variant1_cols = base_cols
    variant2_cols = base_cols + ['is_intra_manhattan', 'dist_intra_manhattan']

    features_choice = skrub.choose_from({
        'traffic_distance_interactions': all_features[variant1_cols],
        'traffic_and_intra_manhattan': all_features[variant2_cols],
    }, name='traffic_features')
    features = skrub.as_data_op(features_choice).skb.set_name('features')

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1800,
        learning_rate=0.022,
        num_leaves=95,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )
    xgb_model = xgb.XGBRegressor(
        n_estimators=1800,
        learning_rate=0.022,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda',
        random_state=42,
    )

    model = VotingRegressor(
        estimators=[('lgb', lgb_model), ('xgb', xgb_model)],
        weights=[0.4, 0.6],
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}