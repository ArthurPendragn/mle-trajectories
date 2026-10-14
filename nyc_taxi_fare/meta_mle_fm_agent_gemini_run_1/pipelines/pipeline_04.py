import lightgbm as lgb


def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]

    year = X["pickup_datetime"].str.slice(0, 4).astype("int32")
    month = X["pickup_datetime"].str.slice(5, 7).astype("int32")
    day = X["pickup_datetime"].str.slice(8, 10).astype("int32")
    hour = X["pickup_datetime"].str.slice(11, 13).astype("int32")

    month_offset = month.map(
        {1: 0, 2: 3, 3: 2, 4: 5, 5: 0, 6: 3, 7: 5, 8: 1, 9: 4, 10: 6, 11: 2, 12: 4}
    )
    y_adj = year - (month < 3).fillna(False).astype("int32")
    day_of_week = (
        y_adj
        + y_adj // 4
        - y_adj // 100
        + y_adj // 400
        + month_offset
        + day
    ) % 7
    is_weekend = ((day_of_week == 0) | (day_of_week == 6)).fillna(False).astype("int32")
    is_weekday = ((day_of_week >= 1) & (day_of_week <= 5)).fillna(False)

    post_2012_hike = (
        (year > 2012) | ((year == 2012) & (month >= 9))
    ).fillna(False).astype("int32")
    is_rush_hour = (
        is_weekday & (hour >= 16) & (hour < 20)
    ).fillna(False).astype("int32")
    is_overnight = ((hour >= 20) | (hour < 6)).fillna(False).astype("int32")

    dlon = X["dropoff_longitude"] - X["pickup_longitude"]
    dlat = X["dropoff_latitude"] - X["pickup_latitude"]
    abs_dlon = dlon.abs()
    abs_dlat = dlat.abs()

    mid_lon = (X["pickup_longitude"] + X["dropoff_longitude"]) * 0.5
    mid_lat = (X["pickup_latitude"] + X["dropoff_latitude"]) * 0.5

    euclidean_dist = ((abs_dlon * 0.75756) ** 2 + abs_dlat**2) ** 0.5 * 111.0
    manhattan_dist = abs_dlon * 84.09 + abs_dlat * 111.0

    p_jfk = (
        ((X["pickup_longitude"] - -73.7781) * 0.75756) ** 2
        + (X["pickup_latitude"] - 40.6413) ** 2
    ) ** 0.5 * 111.0
    d_jfk = (
        ((X["dropoff_longitude"] - -73.7781) * 0.75756) ** 2
        + (X["dropoff_latitude"] - 40.6413) ** 2
    ) ** 0.5 * 111.0
    min_jfk = (p_jfk + d_jfk - (p_jfk - d_jfk).abs()) * 0.5

    p_lga = (
        ((X["pickup_longitude"] - -73.8740) * 0.75756) ** 2
        + (X["pickup_latitude"] - 40.7769) ** 2
    ) ** 0.5 * 111.0
    d_lga = (
        ((X["dropoff_longitude"] - -73.8740) * 0.75756) ** 2
        + (X["dropoff_latitude"] - 40.7769) ** 2
    ) ** 0.5 * 111.0
    min_lga = (p_lga + d_lga - (p_lga - d_lga).abs()) * 0.5

    p_ewr = (
        ((X["pickup_longitude"] - -74.1745) * 0.75756) ** 2
        + (X["pickup_latitude"] - 40.6895) ** 2
    ) ** 0.5 * 111.0
    d_ewr = (
        ((X["dropoff_longitude"] - -74.1745) * 0.75756) ** 2
        + (X["dropoff_latitude"] - 40.6895) ** 2
    ) ** 0.5 * 111.0
    min_ewr = (p_ewr + d_ewr - (p_ewr - d_ewr).abs()) * 0.5

    p_mid = (
        ((X["pickup_longitude"] - -73.9855) * 0.75756) ** 2
        + (X["pickup_latitude"] - 40.7580) ** 2
    ) ** 0.5 * 111.0
    d_mid = (
        ((X["dropoff_longitude"] - -73.9855) * 0.75756) ** 2
        + (X["dropoff_latitude"] - 40.7580) ** 2
    ) ** 0.5 * 111.0
    min_mid = (p_mid + d_mid - (p_mid - d_mid).abs()) * 0.5

    features = (
        X[
            [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
            ]
        ]
        .assign(
            year=year,
            month=month,
            day=day,
            hour=hour,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            post_2012_hike=post_2012_hike,
            is_rush_hour=is_rush_hour,
            is_overnight=is_overnight,
            dlon=dlon,
            dlat=dlat,
            abs_dlon=abs_dlon,
            abs_dlat=abs_dlat,
            mid_lon=mid_lon,
            mid_lat=mid_lat,
            euclidean_km=euclidean_dist,
            manhattan_km=manhattan_dist,
            p_jfk_km=p_jfk,
            d_jfk_km=d_jfk,
            min_jfk_km=min_jfk,
            p_lga_km=p_lga,
            d_lga_km=d_lga,
            min_lga_km=min_lga,
            p_ewr_km=p_ewr,
            d_ewr_km=d_ewr,
            min_ewr_km=min_ewr,
            p_mid_km=p_mid,
            d_mid_km=d_mid,
            min_mid_km=min_mid,
        )
        .skb.set_name("features")
    )

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=8,
    )
    pred = features.skb.apply(model, y=y).skb.set_name("model")

    return {"pred": pred, "scoring": setup["scoring"]}