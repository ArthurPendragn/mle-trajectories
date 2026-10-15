import pandas as pd
import skrub
from lightgbm import LGBMRegressor


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']
    scoring = setup['scoring']

    p_lat = X['pickup_latitude']
    p_lon = X['pickup_longitude']
    d_lat = X['dropoff_latitude']
    d_lon = X['dropoff_longitude']

    d_lat_abs = (d_lat - p_lat).abs()
    d_lon_abs = (d_lon - p_lon).abs()
    d_lat_signed = d_lat - p_lat
    d_lon_signed = d_lon - p_lon

    lat_km = d_lat_abs * 111.0
    lon_km = d_lon_abs * 84.3
    euclidean_km = ((lat_km ** 2) + (lon_km ** 2)) ** 0.5
    manhattan_km = lat_km + lon_km

    # Proximity to major NYC transit hubs (JFK, LGA, EWR, Midtown)
    p_jfk = (((p_lat - 40.6413) * 111.0) ** 2 + ((p_lon - (-73.7781)) * 84.3) ** 2) ** 0.5
    d_jfk = (((d_lat - 40.6413) * 111.0) ** 2 + ((d_lon - (-73.7781)) * 84.3) ** 2) ** 0.5

    p_lga = (((p_lat - 40.7769) * 111.0) ** 2 + ((p_lon - (-73.8740)) * 84.3) ** 2) ** 0.5
    d_lga = (((d_lat - 40.7769) * 111.0) ** 2 + ((d_lon - (-73.8740)) * 84.3) ** 2) ** 0.5

    p_ewr = (((p_lat - 40.6895) * 111.0) ** 2 + ((p_lon - (-74.1745)) * 84.3) ** 2) ** 0.5
    d_ewr = (((d_lat - 40.6895) * 111.0) ** 2 + ((d_lon - (-74.1745)) * 84.3) ** 2) ** 0.5

    p_midtown = (((p_lat - 40.7580) * 111.0) ** 2 + ((p_lon - (-73.9855)) * 84.3) ** 2) ** 0.5
    d_midtown = (((d_lat - 40.7580) * 111.0) ** 2 + ((d_lon - (-73.9855)) * 84.3) ** 2) ** 0.5

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, utc=True)
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    hour = dt.dt.hour
    weekday = dt.dt.dayofweek

    features = X[[
        'passenger_count',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
    ]].assign(
        d_lat_abs=d_lat_abs,
        d_lon_abs=d_lon_abs,
        d_lat=d_lat_signed,
        d_lon=d_lon_signed,
        lat_km=lat_km,
        lon_km=lon_km,
        euclidean_km=euclidean_km,
        manhattan_km=manhattan_km,
        p_jfk=p_jfk,
        d_jfk=d_jfk,
        p_lga=p_lga,
        d_lga=d_lga,
        p_ewr=p_ewr,
        d_ewr=d_ewr,
        p_midtown=p_midtown,
        d_midtown=d_midtown,
        year=year,
        month=month,
        day=day,
        hour=hour,
        weekday=weekday,
    ).skb.set_name('features')

    model = skrub.choose_from(
        {
            'lgbm_leaves31': LGBMRegressor(
                n_estimators=300,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                n_jobs=8,
            ),
            'lgbm_leaves63': LGBMRegressor(
                n_estimators=300,
                learning_rate=0.08,
                num_leaves=63,
                random_state=42,
                n_jobs=8,
            ),
        },
        name='model_variant',
    )

    pred = features.skb.apply(model, y=y).skb.set_name('model')

    return {'pred': pred, 'scoring': scoring}