from lightgbm import LGBMRegressor


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    coords = X[
        [
            'pickup_longitude',
            'pickup_latitude',
            'dropoff_longitude',
            'dropoff_latitude',
            'passenger_count',
        ]
    ]

    d_lat = (X['dropoff_latitude'] - X['pickup_latitude']).abs()
    d_lon = (X['dropoff_longitude'] - X['pickup_longitude']).abs()

    d_lat_km = d_lat * 111.13
    d_lon_km = d_lon * 84.18
    distance_km = (d_lat_km**2 + d_lon_km**2) ** 0.5
    manhattan_km = d_lat_km + d_lon_km

    jfk_dist = (
        ((X['dropoff_latitude'] - 40.6413) * 111.13) ** 2
        + ((X['dropoff_longitude'] - (-73.7781)) * 84.18) ** 2
    ) ** 0.5
    lga_dist = (
        ((X['dropoff_latitude'] - 40.7769) * 111.13) ** 2
        + ((X['dropoff_longitude'] - (-73.8740)) * 84.18) ** 2
    ) ** 0.5
    ewr_dist = (
        ((X['dropoff_latitude'] - 40.6895) * 111.13) ** 2
        + ((X['dropoff_longitude'] - (-74.1745)) * 84.18) ** 2
    ) ** 0.5

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    dayofweek = dt.dt.dayofweek
    hour = dt.dt.hour

    features = coords.assign(
        d_lat=d_lat,
        d_lon=d_lon,
        distance_km=distance_km,
        manhattan_km=manhattan_km,
        jfk_dist=jfk_dist,
        lga_dist=lga_dist,
        ewr_dist=ewr_dist,
        year=year,
        month=month,
        day=day,
        dayofweek=dayofweek,
        hour=hour,
    ).skb.set_name('features')

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.08,
        num_leaves=63,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=42,
        n_jobs=8,
        verbose=-1,
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': scoring}