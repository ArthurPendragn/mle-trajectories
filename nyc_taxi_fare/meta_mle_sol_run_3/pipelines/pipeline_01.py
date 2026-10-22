from xgboost import XGBRegressor


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    coordinates = X[
        ['pickup_longitude', 'pickup_latitude',
         'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    ].astype('float32').skb.set_name('raw_coordinates')

    timestamp = X['pickup_datetime'].skb.apply_func(
        pd.to_datetime, utc=True, errors='coerce'
    )
    dates = coordinates[['passenger_count']].assign(
        year=timestamp.dt.year,
        month=timestamp.dt.month,
        day=timestamp.dt.day,
        weekday=timestamp.dt.dayofweek,
        hour=timestamp.dt.hour,
        minute=timestamp.dt.minute,
        second=timestamp.dt.second,
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('dates')

    dx = (
        X['dropoff_longitude'] - X['pickup_longitude']
    ) * 84.3
    dy = (
        X['dropoff_latitude'] - X['pickup_latitude']
    ) * 111.2

    geometry = coordinates[['passenger_count']].assign(
        delta_x=dx,
        delta_y=dy,
        absolute_x=dx.abs(),
        absolute_y=dy.abs(),
        euclidean_distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan_distance=dx.abs() + dy.abs(),
        midpoint_longitude=(
            X['pickup_longitude'] + X['dropoff_longitude']
        ) / 2,
        midpoint_latitude=(
            X['pickup_latitude'] + X['dropoff_latitude']
        ) / 2,
    ).drop(columns=['passenger_count']).astype('float32').skb.set_name('geometry')

    features = coordinates.skb.concat([dates, geometry], axis=1)
    features = features.replace(
        [float('inf'), -float('inf')], float('nan')
    ).skb.set_name('finite_features')

    model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        n_estimators=700,
        max_depth=8,
        learning_rate=0.06,
        min_child_weight=20,
        reg_lambda=10,
        max_bin=256,
        random_state=714,
        n_jobs=16,
    )
    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {
        'pred': pred,
        'scoring': setup['scoring'],
        'row_keys': setup['row_keys'],
    }