import numpy as np
import pandas as pd
import skrub
from xgboost import XGBRegressor

def build():
    setup = build_evaluation()
    X = setup["X"]
    y = setup["y"]

    dt = X["pickup_datetime"].skb.apply_func(pd.to_datetime, utc=True, errors="coerce")
    calendar = X[["passenger_count"]].assign(
        year=dt.dt.year,
        month=dt.dt.month,
        day=dt.dt.day,
        weekday=dt.dt.dayofweek,
        hour=dt.dt.hour,
        minute=dt.dt.minute,
    ).astype("float32").skb.set_name("calendar")

    dx = (X["dropoff_longitude"] - X["pickup_longitude"]) * 84.4
    dy = (X["dropoff_latitude"] - X["pickup_latitude"]) * 111.2
    geography = X[[
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
    ]].assign(
        dx=dx,
        dy=dy,
        distance=(dx ** 2 + dy ** 2) ** 0.5,
        manhattan=dx.abs() + dy.abs(),
    ).astype("float32").skb.set_name("geography")

    features = geography.skb.concat([calendar], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).skb.set_name("finite_features")

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_estimators=600,
        max_depth=8,
        learning_rate=0.06,
        min_child_weight=20,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=10,
        n_jobs=8,
        random_state=1971,
    )
    pred = features.skb.apply(model, y=y).skb.set_name("model")
    result = {"pred": pred, "scoring": setup["scoring"]}
    if "row_keys" in setup:
        result["row_keys"] = setup["row_keys"]
    return result