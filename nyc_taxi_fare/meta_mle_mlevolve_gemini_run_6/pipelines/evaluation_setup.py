import pandas as pd
from sklearn.model_selection import KFold
import skrub

TRAIN_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"


def build():
    # Subsampling rationale:
    # Using 1,000,000 rows deterministically read from the 55.4M training dataset.
    # This provides 100x the sample size of the test set (9,914 rows), giving
    # reliable evaluation while allowing rapid cross-validation within the execution timeout.
    train = skrub.as_data_op(TRAIN_PATH).skb.apply_func(
        pd.read_csv,
        nrows=1000000,
    )

    # Filter out corrupted records (non-positive fares, coordinates outside NYC metro area,
    # passenger counts outside [1, 6]), matching the domain boundaries observed in the test set.
    valid_mask = (
        (train["fare_amount"] > 0)
        & (train["fare_amount"] <= 500.0)
        & (train["pickup_longitude"] >= -75.0)
        & (train["pickup_longitude"] <= -72.0)
        & (train["pickup_latitude"] >= 40.0)
        & (train["pickup_latitude"] <= 42.0)
        & (train["dropoff_longitude"] >= -75.0)
        & (train["dropoff_longitude"] <= -72.0)
        & (train["dropoff_latitude"] >= 40.0)
        & (train["dropoff_latitude"] <= 42.0)
        & (train["passenger_count"] >= 1)
        & (train["passenger_count"] <= 6)
    ).fillna(False)

    train_clean = train[valid_mask].reset_index(drop=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    split_kwargs = {}

    y = train_clean["fare_amount"].skb.mark_as_y()
    X = train_clean.drop(columns=["fare_amount"]).skb.mark_as_X(
        cv=cv,
        split_kwargs=split_kwargs,
    )

    return {
        "X": X,
        "y": y,
        "scoring": "neg_root_mean_squared_error",
    }


build_evaluation = build