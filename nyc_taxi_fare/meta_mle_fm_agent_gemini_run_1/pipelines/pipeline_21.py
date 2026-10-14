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

    month_offset = month.map(
        {1: 0, 2: 3, 3: 2, 4: 5, 5: 0, 6: 3, 7: 5, 8: 1, 9: 4, 10: 6, 11: 2, 12: 4}
    )
    y_adj = year - (month < 3).fillna(False).astype('int32')
    day_of_week = (
        y_adj
        + y_adj // 4
        - y_adj // 100
        + y_adj // 400
        + month_offset
        + day
    ) % 7

    is_weekend = ((day_of_week == 0) | (day_of_week == 6)).fillna(False).astype(
        'int32'
    )
    is_weekday = ((day_of_week >= 1) & (day_of_week <= 5)).fillna(False)

    post_2012_hike = (
        ((year > 2012) | (year == 2012) & (month >= 9))
        .fillna(False)
        .astype('int32')
    )
    is_rush_hour = (is_weekday & (hour >= 16) & (hour < 20)).fillna(
        False
    ).astype('int32')
    is_overnight = ((hour >= 20) | (hour < 6)).fillna(False).astype('int32')

    abs_dlon = (X['dropoff_longitude'] - X['pickup_longitude']).abs()
    abs_dlat = (X['dropoff_latitude'] - X['pickup_latitude']).abs()

    euclidean_dist = (
        ((abs_dlon * 0.75756) ** 2 + abs_dlat**2) ** 0.5 * 111.0
    )
    manhattan_dist = abs_dlon * 84.09 + abs_dlat * 111.0
    sqrt_euclidean = euclidean_dist**0.5
    sqrt_manhattan = manhattan_dist**0.5
    log_euclidean = (euclidean_dist + 1.0).skb.apply_func(np.log)
    log_manhattan = (manhattan_dist + 1.0).skb.apply_func(np.log)

    p_jfk = (
        ((X['pickup_longitude'] - -73.7781) * 0.75756) ** 2
        + (X['pickup_latitude'] - 40.6413) ** 2
    ) ** 0.5 * 111.0
    d_jfk = (
        ((X['dropoff_longitude'] - -73.7781) * 0.75756) ** 2
        + (X['dropoff_latitude'] - 40.6413) ** 2
    ) ** 0.5 * 111.0
    is_jfk = ((p_jfk < 3.0) | (d_jfk < 3.0)).fillna(False).astype('int32')

    p_lga = (
        ((X['pickup_longitude'] - -73.874) * 0.75756) ** 2
        + (X['pickup_latitude'] - 40.7769) ** 2
    ) ** 0.5 * 111.0
    d_lga = (
        ((X['dropoff_longitude'] - -73.874) * 0.75756) ** 2
        + (X['dropoff_latitude'] - 40.7769) ** 2
    ) ** 0.5 * 111.0
    is_lga = ((p_lga < 3.0) | (d_lga < 3.0)).fillna(False).astype('int32')

    p_ewr = (
        ((X['pickup_longitude'] - -74.1745) * 0.75756) ** 2
        + (X['pickup_latitude'] - 40.6895) ** 2
    ) ** 0.5 * 111.0
    d_ewr = (
        ((X['dropoff_longitude'] - -74.1745) * 0.75756) ** 2
        + (X['dropoff_latitude'] - 40.6895) ** 2
    ) ** 0.5 * 111.0
    is_ewr = ((p_ewr < 3.0) | (d_ewr < 3.0)).fillna(False).astype('int32')

    p_mid = (
        ((X['pickup_longitude'] - -73.9855) * 0.75756) ** 2
        + (X['pickup_latitude'] - 40.758) ** 2
    ) ** 0.5 * 111.0
    d_mid = (
        ((X['dropoff_longitude'] - -73.9855) * 0.75756) ** 2
        + (X['dropoff_latitude'] - 40.758) ** 2
    ) ** 0.5 * 111.0

    euc_post2012 = euclidean_dist * post_2012_hike
    man_post2012 = manhattan_dist * post_2012_hike

    features = (
        X[[
            'pickup_longitude',
            'pickup_latitude',
            'dropoff_longitude',
            'dropoff_latitude',
            'passenger_count',
        ]]
        .assign(
            year=year,
            month=month,
            day=day,
            hour=hour,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            post_2012_hike=post_2012_hike,
            is_rush_hour=is_rush_hour,
            is_overnight=is_overnight,
            abs_dlon=abs_dlon,
            abs_dlat=abs_dlat,
            euclidean_km=euclidean_dist,
            manhattan_km=manhattan_dist,
            sqrt_euclidean=sqrt_euclidean,
            sqrt_manhattan=sqrt_manhattan,
            log_euclidean=log_euclidean,
            log_manhattan=log_manhattan,
            p_jfk_km=p_jfk,
            d_jfk_km=d_jfk,
            is_jfk=is_jfk,
            p_lga_km=p_lga,
            d_lga_km=d_lga,
            is_lga=is_lga,
            p_ewr_km=p_ewr,
            d_ewr_km=d_ewr,
            is_ewr=is_ewr,
            p_mid_km=p_mid,
            d_mid_km=d_mid,
            euc_post2012=euc_post2012,
            man_post2012=man_post2012,
        )
        .skb.set_name('features')
    )

    lgb_est_63 = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )
    lgb_est_95 = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=95,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )
    xgb_est = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda',
        random_state=42,
    )

    model_63 = VotingRegressor(
        estimators=[('lgb', lgb_est_63), ('xgb', xgb_est)], weights=[0.5, 0.5]
    )
    model_95 = VotingRegressor(
        estimators=[('lgb', lgb_est_95), ('xgb', xgb_est)], weights=[0.5, 0.5]
    )

    model = skrub.choose_from(
        {'leaves63_depth8': model_63, 'leaves95_depth8': model_95},
        name='ensemble_capacity',
    )
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}