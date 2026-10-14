import pandas as pd
from sklearn.model_selection import KFold
import skrub

TRAIN_SOURCE = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
SAMPLE_SIZE = 1_000_000


def build():
    train_raw = skrub.as_data_op(TRAIN_SOURCE).skb.apply_func(
        pd.read_csv, nrows=SAMPLE_SIZE
    )

    mask = (
        (train_raw["fare_amount"] > 0)
        & (train_raw["fare_amount"] <= 500)
        & (train_raw["pickup_longitude"] >= -75.0)
        & (train_raw["pickup_longitude"] <= -72.0)
        & (train_raw["pickup_latitude"] >= 40.0)
        & (train_raw["pickup_latitude"] <= 42.0)
        & (train_raw["dropoff_longitude"] >= -75.0)
        & (train_raw["dropoff_longitude"] <= -72.0)
        & (train_raw["dropoff_latitude"] >= 40.0)
        & (train_raw["dropoff_latitude"] <= 42.0)
        & (train_raw["passenger_count"] >= 1)
        & (train_raw["passenger_count"] <= 6)
    ).fillna(False)

    train_clean = train_raw[mask].reset_index(drop=True)

    y = train_clean["fare_amount"]
    X = train_clean.drop(columns=["fare_amount"])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    X_marked = X.skb.mark_as_X(cv=cv, split_kwargs={})
    y_marked = y.skb.mark_as_y()

    return {
        "X": X_marked,
        "y": y_marked,
        "scoring": "neg_root_mean_squared_error",
    }