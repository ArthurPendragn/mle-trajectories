import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor

def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True, errors='coerce')
    calendar = X[['passenger_count']].assign(
        year=dt.dt.year, month=dt.dt.month, day=dt.dt.day,
        weekday=dt.dt.dayofweek, hour=dt.dt.hour, minute=dt.dt.minute
    ).astype('float32').skb.set_name('calendar')
    dx = (X['dropoff_longitude'] - X['pickup_longitude']) * 84.4
    dy = (X['dropoff_latitude'] - X['pickup_latitude']) * 111.2
    geography = X[[
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude'
    ]].assign(
        dx=dx, dy=dy,
        distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs()
    ).astype('float32').skb.set_name('geography')
    features = geography.skb.concat([calendar], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).skb.set_name('finite_features')
    model = CatBoostRegressor(
        task_type='GPU',
        devices='0',
        loss_function='RMSE',
        iterations=600,
        depth=8,
        learning_rate=0.08,
        l2_leaf_reg=10,
        random_seed=1971,
        thread_count=8,
        allow_writing_files=False,
        verbose=False
    )
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    result = {'pred': pred, 'scoring': setup['scoring']}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result