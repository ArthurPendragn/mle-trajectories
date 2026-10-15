import pandas as pd
import skrub
from sklearn.model_selection import KFold

TRAIN_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
TEST_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/test.csv"
SAMPLE_SUBMISSION_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/sample_submission.csv"

N_SAMPLES = 1_000_000
N_SPLITS = 5
RANDOM_STATE = 42
SCORING = "neg_root_mean_squared_error"


def mark_X(df, cv, split_kwargs=None):
    if split_kwargs is None:
        split_kwargs = {}
    if hasattr(df.skb, "mark_as_X"):
        return df.skb.mark_as_X(cv=cv, split_kwargs=split_kwargs)
    elif hasattr(skrub, "mark_as_X"):
        return skrub.mark_as_X(df, cv=cv, split_kwargs=split_kwargs)
    raise AttributeError("mark_as_X could not be resolved on DataOp or skrub module")


def mark_y(s):
    if hasattr(s.skb, "mark_as_y"):
        return s.skb.mark_as_y()
    elif hasattr(skrub, "mark_as_y"):
        return skrub.mark_as_y(s)
    raise AttributeError("mark_as_y could not be resolved on DataOp or skrub module")


def build_evaluation():
    train_raw = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv, nrows=N_SAMPLES)
    test_raw = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    valid_fare = ((train_raw["fare_amount"] >= 2.5) & (train_raw["fare_amount"] <= 500.0)).fillna(False)
    valid_pickup_lon = ((train_raw["pickup_longitude"] >= -75.0) & (train_raw["pickup_longitude"] <= -72.0)).fillna(False)
    valid_pickup_lat = ((train_raw["pickup_latitude"] >= 40.0) & (train_raw["pickup_latitude"] <= 42.0)).fillna(False)
    valid_dropoff_lon = ((train_raw["dropoff_longitude"] >= -75.0) & (train_raw["dropoff_longitude"] <= -72.0)).fillna(False)
    valid_dropoff_lat = ((train_raw["dropoff_latitude"] >= 40.0) & (train_raw["dropoff_latitude"] <= 42.0)).fillna(False)
    valid_passengers = ((train_raw["passenger_count"] >= 1) & (train_raw["passenger_count"] <= 6)).fillna(False)

    valid_mask = (
        valid_fare
        & valid_pickup_lon
        & valid_pickup_lat
        & valid_dropoff_lon
        & valid_dropoff_lat
        & valid_passengers
    )

    clean_train = train_raw[valid_mask].reset_index(drop=True)

    raw_X = clean_train.drop(columns=["fare_amount"])
    raw_y = clean_train["fare_amount"]
    row_keys = clean_train["key"]

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X = mark_X(raw_X, cv=cv, split_kwargs={})
    y = mark_y(raw_y)

    audit = {
        "clean_train_summary": clean_train[["fare_amount", "passenger_count"]].describe(),
        "test_passenger_summary": test_raw[["passenger_count"]].describe(),
    }

    return {
        "X": X,
        "y": y,
        "scoring": SCORING,
        "row_keys": row_keys,
        "audit": audit,
    }


def build():
    return build_evaluation()