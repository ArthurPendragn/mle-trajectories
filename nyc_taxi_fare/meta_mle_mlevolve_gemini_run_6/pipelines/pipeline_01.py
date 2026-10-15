from lightgbm import LGBMRegressor
import pandas as pd
import skrub


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime)

    features = X.assign(
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        dayofweek=dt.dt.dayofweek,
        hour=dt.dt.hour,
        abs_lat_diff=(X['dropoff_latitude'] - X['pickup_latitude']).abs(),
        abs_lon_diff=(X['dropoff_longitude'] - X['pickup_longitude']).abs(),
        dist_euclidean_km=(
            ((X['dropoff_latitude'] - X['pickup_latitude']) * 111.0) ** 2
            + ((X['dropoff_longitude'] - X['pickup_longitude']) * 84.11) ** 2
        ) ** 0.5,
        dist_manhattan_km=(
            (X['dropoff_latitude'] - X['pickup_latitude']).abs() * 111.0
            + (X['dropoff_longitude'] - X['pickup_longitude']).abs() * 84.11
        ),
        jfk_drop_dist=(
            ((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2
            + ((X['dropoff_longitude'] - -73.7781) * 84.11) ** 2
        ) ** 0.5,
        jfk_pick_dist=(
            ((X['pickup_latitude'] - 40.6413) * 111.0) ** 2
            + ((X['pickup_longitude'] - -73.7781) * 84.11) ** 2
        ) ** 0.5,
        lga_drop_dist=(
            ((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2
            + ((X['dropoff_longitude'] - -73.8740) * 84.11) ** 2
        ) ** 0.5,
        lga_pick_dist=(
            ((X['pickup_latitude'] - 40.7769) * 111.0) ** 2
            + ((X['pickup_longitude'] - -73.8740) * 84.11) ** 2
        ) ** 0.5,
        ewr_drop_dist=(
            ((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2
            + ((X['dropoff_longitude'] - -74.1745) * 84.11) ** 2
        ) ** 0.5,
        ewr_pick_dist=(
            ((X['pickup_latitude'] - 40.6895) * 111.0) ** 2
            + ((X['pickup_longitude'] - -74.1745) * 84.11) ** 2
        ) ** 0.5,
    ).drop(columns=['key', 'pickup_datetime']).skb.set_name('spatial_temporal_features')

    model = LGBMRegressor(
        objective='regression',
        n_estimators=skrub.choose_from({'trees_300': 300, 'trees_600': 600}, name='n_estimators'),
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': setup['scoring']}