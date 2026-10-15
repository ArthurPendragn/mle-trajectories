import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek
    hour = dt.dt.hour

    d_lat = (X['dropoff_latitude'] - X['pickup_latitude']).abs()
    d_lon = (X['dropoff_longitude'] - X['pickup_longitude']).abs()

    # Flat-earth approximation for NYC (lat ~ 40.75 deg):
    # 1 deg lat ~ 111.0 km, 1 deg lon ~ 84.3 km
    dist_km = (((d_lat * 111.0) ** 2) + ((d_lon * 84.3) ** 2)).skb.apply_func(np.sqrt)

    # Distances to major airports (JFK, LGA, EWR)
    # JFK: (40.6413, -73.7781)
    jfk_pick_dist = (
        ((X['pickup_latitude'] - 40.6413) * 111.0) ** 2
        + ((X['pickup_longitude'] - (-73.7781)) * 84.3) ** 2
    ).skb.apply_func(np.sqrt)
    jfk_drop_dist = (
        ((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2
        + ((X['dropoff_longitude'] - (-73.7781)) * 84.3) ** 2
    ).skb.apply_func(np.sqrt)

    # LGA: (40.7769, -73.8740)
    lga_pick_dist = (
        ((X['pickup_latitude'] - 40.7769) * 111.0) ** 2
        + ((X['pickup_longitude'] - (-73.8740)) * 84.3) ** 2
    ).skb.apply_func(np.sqrt)
    lga_drop_dist = (
        ((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2
        + ((X['dropoff_longitude'] - (-73.8740)) * 84.3) ** 2
    ).skb.apply_func(np.sqrt)

    # EWR: (40.6895, -74.1745)
    ewr_pick_dist = (
        ((X['pickup_latitude'] - 40.6895) * 111.0) ** 2
        + ((X['pickup_longitude'] - (-74.1745)) * 84.3) ** 2
    ).skb.apply_func(np.sqrt)
    ewr_drop_dist = (
        ((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2
        + ((X['dropoff_longitude'] - (-74.1745)) * 84.3) ** 2
    ).skb.apply_func(np.sqrt)

    features = (
        X[[
            'pickup_longitude',
            'pickup_latitude',
            'dropoff_longitude',
            'dropoff_latitude',
            'passenger_count',
        ]]
        .assign(
            d_lat=d_lat,
            d_lon=d_lon,
            dist_km=dist_km,
            jfk_pick_dist=jfk_pick_dist,
            jfk_drop_dist=jfk_drop_dist,
            lga_pick_dist=lga_pick_dist,
            lga_drop_dist=lga_drop_dist,
            ewr_pick_dist=ewr_pick_dist,
            ewr_drop_dist=ewr_drop_dist,
            year=year,
            month=month,
            day=day,
            dayofweek=dayofweek,
            hour=hour,
        )
        .skb.set_name('features')
    )

    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=8,
        verbosity=-1,
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys'],
    }