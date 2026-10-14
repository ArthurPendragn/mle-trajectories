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

    p_jfk = (((X['pickup_longitude'] - (-73.7781)) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.6413) ** 2) ** 0.5 * 111.0
    d_jfk = (((X['dropoff_longitude'] - (-73.7781)) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.6413) ** 2) ** 0.5 * 111.0

    p_lga = (((X['pickup_longitude'] - (-73.8740)) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.7769) ** 2) ** 0.5 * 111.0
    d_lga = (((X['dropoff_longitude'] - (-73.8740)) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.7769) ** 2) ** 0.5 * 111.0

    p_ewr = (((X['pickup_longitude'] - (-74.1745)) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.6895) ** 2) ** 0.5 * 111.0
    d_ewr = (((X['dropoff_longitude'] - (-74.1745)) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.6895) ** 2) ** 0.5 * 111.0

    p_mid = (((X['pickup_longitude'] - (-73.9855)) * 0.75756) ** 2 + (X['pickup_latitude'] - 40.7580) ** 2) ** 0.5 * 111.0
    d_mid = (((X['dropoff_longitude'] - (-73.9855)) * 0.75756) ** 2 + (X['dropoff_latitude'] - 40.7580) ** 2) ** 0.5 * 111.0

    features = X[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
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
        p_jfk_km=p_jfk,
        d_jfk_km=d_jfk,
        p_lga_km=p_lga,
        d_lga_km=d_lga,
        p_ewr_km=p_ewr,
        d_ewr_km=d_ewr,
        p_mid_km=p_mid,
        d_mid_km=d_mid,
    ).skb.set_name('features')

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

    xgb_est_11 = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=11,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda',
        random_state=42,
    )

    xgb_choice = skrub.choose_from({
        'xgb_depth_10': xgb_est_10,
        'xgb_depth_11': xgb_est_11,
    }, name='xgb_depth')

    model = VotingRegressor(
        estimators=[('lgb', lgb_est), ('xgb', xgb_choice)],
        weights=[0.5, 0.5],
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}