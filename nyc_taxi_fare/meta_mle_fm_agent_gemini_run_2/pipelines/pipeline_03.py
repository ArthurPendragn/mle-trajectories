from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build():
    setup = build_evaluation()
    X = setup['X']
    y = setup['y']

    dt = X['pickup_datetime'].skb.apply_func(pd.to_datetime, format='%Y-%m-%d %H:%M:%S UTC')
    year = dt.dt.year.astype('int32')
    month = dt.dt.month.astype('int32')
    day = dt.dt.day.astype('int32')
    hour = dt.dt.hour.astype('int32')
    weekday = dt.dt.dayofweek.astype('int32')

    d_lon = (X['dropoff_longitude'] - X['pickup_longitude']).abs()
    d_lat = (X['dropoff_latitude'] - X['pickup_latitude']).abs()
    euclidean_dist = ((d_lat * 111.0) ** 2 + (d_lon * 84.0) ** 2) ** 0.5

    p_lat_rad = X['pickup_latitude'] * (np.pi / 180.0)
    d_lat_rad = X['dropoff_latitude'] * (np.pi / 180.0)
    d_lon_rad = (X['dropoff_longitude'] - X['pickup_longitude']) * (np.pi / 180.0)
    d_lat_diff = (X['dropoff_latitude'] - X['pickup_latitude']) * (np.pi / 180.0)

    sin_dlat = (d_lat_diff * 0.5).skb.apply_func(np.sin)
    sin_dlon = (d_lon_rad * 0.5).skb.apply_func(np.sin)
    cos_plat = p_lat_rad.skb.apply_func(np.cos)
    cos_dlat = d_lat_rad.skb.apply_func(np.cos)
    a = sin_dlat * sin_dlat + cos_plat * cos_dlat * sin_dlon * sin_dlon
    a_clipped = a.skb.apply_func(np.clip, 0.0, 1.0)
    haversine_dist = a_clipped.skb.apply_func(np.sqrt).skb.apply_func(np.arcsin) * (2.0 * 6371.0)

    dist_pickup_jfk = (((X['pickup_latitude'] - 40.6413) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.7781) * 84.0) ** 2) ** 0.5
    dist_dropoff_jfk = (((X['dropoff_latitude'] - 40.6413) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.7781) * 84.0) ** 2) ** 0.5
    dist_pickup_lga = (((X['pickup_latitude'] - 40.7769) * 111.0) ** 2 + ((X['pickup_longitude'] - -73.874) * 84.0) ** 2) ** 0.5
    dist_dropoff_lga = (((X['dropoff_latitude'] - 40.7769) * 111.0) ** 2 + ((X['dropoff_longitude'] - -73.874) * 84.0) ** 2) ** 0.5
    dist_pickup_ewr = (((X['pickup_latitude'] - 40.6895) * 111.0) ** 2 + ((X['pickup_longitude'] - -74.1745) * 84.0) ** 2) ** 0.5
    dist_dropoff_ewr = (((X['dropoff_latitude'] - 40.6895) * 111.0) ** 2 + ((X['dropoff_longitude'] - -74.1745) * 84.0) ** 2) ** 0.5

    features = X.drop(columns=['key', 'pickup_datetime']).assign(
        abs_diff_lon=d_lon,
        abs_diff_lat=d_lat,
        euclidean_dist=euclidean_dist,
        haversine_dist=haversine_dist,
        dist_pickup_jfk=dist_pickup_jfk,
        dist_dropoff_jfk=dist_dropoff_jfk,
        dist_pickup_lga=dist_pickup_lga,
        dist_dropoff_lga=dist_dropoff_lga,
        dist_pickup_ewr=dist_pickup_ewr,
        dist_dropoff_ewr=dist_dropoff_ewr,
        year=year,
        month=month,
        day=day,
        hour=hour,
        weekday=weekday,
    ).skb.set_name('features')

    model = skrub.choose_from({
        'ridge_alpha_1': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
        'ridge_alpha_10': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))]),
        'ridge_alpha_100': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=100.0))]),
    }, name='model_variant')

    pred = features.skb.apply(model, y=y).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}