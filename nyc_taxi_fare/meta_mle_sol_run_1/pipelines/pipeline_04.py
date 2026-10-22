import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor


def ride_features(X):
    coordinates = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    raw = X[coordinates + ['passenger_count']].replace([np.inf, -np.inf], np.nan)
    flags = {}
    clean = {}
    for endpoint in ['pickup', 'dropoff']:
        lon = raw[endpoint + '_longitude']
        lat = raw[endpoint + '_latitude']
        valid = (lon.between(-180, 180) & lat.between(-90, 90)).fillna(False)
        local = (lon.between(-75, -72) & lat.between(40, 42)).fillna(False)
        flags[endpoint + '_geographic_valid'] = valid.astype('float32')
        flags[endpoint + '_nyc_region'] = local.astype('float32')
        clean[endpoint + '_longitude'] = lon.where(valid)
        clean[endpoint + '_latitude'] = lat.where(valid)
        for axis in ['longitude', 'latitude']:
            col = endpoint + '_' + axis
            flags[col + '_missing'] = raw[col].isna().astype('float32')
    location = raw.assign(**flags).astype('float32').skb.set_name('locations')
    east = ((clean['dropoff_longitude'] - clean['pickup_longitude']) * 84.3).clip(-300, 300)
    north = ((clean['dropoff_latitude'] - clean['pickup_latitude']) * 111.32).clip(-300, 300)
    distance = X[['passenger_count']].assign(
        east_km=east,
        north_km=north,
        abs_east_km=east.abs(),
        abs_north_km=north.abs(),
        euclidean_km=(east ** 2 + north ** 2).skb.apply_func(np.sqrt),
        manhattan_km=east.abs() + north.abs()
    ).drop(columns=['passenger_count'])
    airport_features = {}
    for airport, lon, lat in [
        ('jfk', -73.7781, 40.6413),
        ('lga', -73.874, 40.7769),
        ('ewr', -74.1745, 40.6895)
    ]:
        for endpoint in ['pickup', 'dropoff']:
            dx = ((clean[endpoint + '_longitude'] - lon) * 84.3).clip(-300, 300)
            dy = ((clean[endpoint + '_latitude'] - lat) * 111.32).clip(-300, 300)
            airport_features[endpoint + '_' + airport + '_km'] = (
                dx ** 2 + dy ** 2
            ).skb.apply_func(np.sqrt)
    distance = distance.assign(**airport_features).replace(
        [np.inf, -np.inf], np.nan
    ).astype('float32').skb.set_name('ride_geometry')
    timestamp = X['pickup_datetime'].skb.apply_func(
        pd.to_datetime, utc=True, errors='coerce'
    )
    dates = X[['passenger_count']].assign(
        year=timestamp.dt.year,
        month=timestamp.dt.month,
        day=timestamp.dt.day,
        weekday=timestamp.dt.dayofweek,
        hour=timestamp.dt.hour,
        elapsed_days=(timestamp - pd.Timestamp('2009-01-01', tz='UTC')).dt.days
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('calendar')
    return location.skb.concat([distance, dates], axis=1)


def build():
    setup = build_evaluation()
    features = ride_features(setup['X'])
    models = {}
    for objective in ['RMSE', 'Huber:delta=10']:
        models[objective] = CatBoostRegressor(
            loss_function=objective,
            task_type='GPU',
            devices='0',
            iterations=400,
            depth=6,
            learning_rate=0.08,
            random_seed=73191,
            thread_count=8,
            verbose=False,
            allow_writing_files=False
        )
    model = skrub.choose_from(models, name='training_objective')
    pred = features.skb.apply(model, y=setup['y']).skb.set_name('model')
    return {
        'pred': pred,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys']
    }