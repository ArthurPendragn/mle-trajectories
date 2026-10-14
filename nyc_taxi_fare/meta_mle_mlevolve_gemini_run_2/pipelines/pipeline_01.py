from lightgbm import LGBMRegressor
import pandas as pd


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    pickup_dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True)
    year = pickup_dt.dt.year
    month = pickup_dt.dt.month
    day = pickup_dt.dt.day
    dayofweek = pickup_dt.dt.dayofweek
    hour = pickup_dt.dt.hour

    d_lat = X['dropoff_latitude'] - X['pickup_latitude']
    d_lon = X['dropoff_longitude'] - X['pickup_longitude']
    abs_d_lat = d_lat.abs()
    abs_d_lon = d_lon.abs()

    # Equirectangular distance approximation for NYC coordinates (~40.75 N)
    dist_km = ((d_lat * 111.0) ** 2 + (d_lon * 84.3) ** 2) ** 0.5
    manhattan_km = abs_d_lat * 111.0 + abs_d_lon * 84.3

    # Distance to major NYC area airports (JFK, LGA, EWR)
    jfk_pickup_dist = (((X['pickup_latitude'] - 40.6413) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.7781) * 84.3) ** 2) ** 0.5
    jfk_dropoff_dist = (((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.7781) * 84.3) ** 2) ** 0.5

    lga_pickup_dist = (((X['pickup_latitude'] - 40.7769) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.8740) * 84.3) ** 2) ** 0.5
    lga_dropoff_dist = (((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.8740) * 84.3) ** 2) ** 0.5

    ewr_pickup_dist = (((X['pickup_latitude'] - 40.6895) * 111.0) ** 2 + ((X['pickup_longitude'] - -74.1745) * 84.3) ** 2) ** 0.5
    ewr_dropoff_dist = (((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2 + ((X['dropoff_longitude'] - -74.1745) * 84.3) ** 2) ** 0.5

    features = X.assign(
        year=year,
        month=month,
        day=day,
        dayofweek=dayofweek,
        hour=hour,
        d_lat=d_lat,
        d_lon=d_lon,
        abs_d_lat=abs_d_lat,
        abs_d_lon=abs_d_lon,
        dist_km=dist_km,
        manhattan_km=manhattan_km,
        jfk_pickup_dist=jfk_pickup_dist,
        jfk_dropoff_dist=jfk_dropoff_dist,
        lga_pickup_dist=lga_pickup_dist,
        lga_dropoff_dist=lga_dropoff_dist,
        ewr_pickup_dist=ewr_pickup_dist,
        ewr_dropoff_dist=ewr_dropoff_dist,
    ).drop(columns=['key', 'pickup_datetime']).skb.set_name('geo_temporal_features')

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )
    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {
        'pred': pred,
        'scoring': setup['scoring'],
    }