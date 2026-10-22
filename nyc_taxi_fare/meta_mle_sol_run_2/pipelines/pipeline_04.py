import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor

def build():
    setup = build_evaluation()
    X, y = setup['X'], setup['y']
    dt = X['pickup_datetime'].skb.apply_func(
        pd.to_datetime, utc=True, errors='coerce'
    )
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
        dx=dx, dy=dy, distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs()
    ).astype('float32').skb.set_name('geography')

    angle = np.deg2rad(29.0)
    c, s = np.cos(angle), np.sin(angle)
    px = (X['pickup_longitude'] + 74.0) * 84.4
    py = (X['pickup_latitude'] - 40.75) * 111.2
    qx = (X['dropoff_longitude'] + 74.0) * 84.4
    qy = (X['dropoff_latitude'] - 40.75) * 111.2
    route = X[['passenger_count']].assign(
        pickup_grid_x=px*c-py*s,
        pickup_grid_y=px*s+py*c,
        dropoff_grid_x=qx*c-qy*s,
        dropoff_grid_y=qx*s+qy*c,
        grid_manhattan=(dx*c-dy*s).abs()+(dx*s+dy*c).abs(),
        bearing=dy.skb.apply_func(np.arctan2, dx),
        midpoint_x=(px+qx)*0.5,
        midpoint_y=(py+qy)*0.5,
        post_tariff=(dt >= pd.Timestamp('2012-09-04', tz='UTC')).fillna(False),
        night=((dt.dt.hour >= 20) | (dt.dt.hour < 6)).fillna(False),
        weekday_rush=((dt.dt.dayofweek < 5) &
                      (dt.dt.hour >= 16) & (dt.dt.hour < 20)).fillna(False)
    ).drop(columns=['passenger_count'])
    extra_columns = [
        'pickup_grid_x', 'pickup_grid_y', 'dropoff_grid_x',
        'dropoff_grid_y', 'grid_manhattan', 'bearing',
        'midpoint_x', 'midpoint_y', 'post_tariff', 'night', 'weekday_rush'
    ]
    for airport, lon, lat in [
        ('jfk', -73.7781, 40.6413),
        ('lga', -73.8740, 40.7769),
        ('newark', -74.1745, 40.6895)
    ]:
        for endpoint in ['pickup', 'dropoff']:
            name = endpoint + '_' + airport + '_distance'
            ax = (X[endpoint + '_longitude'] - lon) * 84.4
            ay = (X[endpoint + '_latitude'] - lat) * 111.2
            route = route.assign(**{name: (ax**2 + ay**2)**0.5})
            extra_columns.append(name)
    route = route.astype('float32').skb.set_name('route_regime')

    basic_columns = [
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'dx', 'dy', 'distance', 'manhattan',
        'passenger_count', 'year', 'month', 'day',
        'weekday', 'hour', 'minute'
    ]
    columns = skrub.choose_from(
        {'basic': basic_columns, 'augmented': basic_columns + extra_columns},
        name='route_regime_features'
    )
    features = geography.skb.concat([calendar, route], axis=1)[columns]
    features = features.replace(
        [np.inf, -np.inf], np.nan
    ).skb.set_name('finite_features')
    model = LGBMRegressor(
        objective='regression', n_estimators=400, num_leaves=31,
        learning_rate=0.07, min_child_samples=100, reg_lambda=10,
        n_jobs=8, random_state=1971, verbosity=-1, device_type='cpu'
    )
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    result = {'pred': pred, 'scoring': setup['scoring']}
    if 'row_keys' in setup:
        result['row_keys'] = setup['row_keys']
    return result