import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

TRAIN_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
TEST_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/test.csv"
SAMPLE_SIZE = 500000


def build():
    # Read a sample of the 55M training rows to allow rapid iteration within the resource budget.
    df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv, nrows=SAMPLE_SIZE)

    # Clean population: realistic NYC geographic bounding box and valid fare/passenger ranges.
    mask = (
        (df["fare_amount"] >= 2.5)
        & (df["fare_amount"] <= 500.0)
        & (df["pickup_longitude"] >= -74.5)
        & (df["pickup_longitude"] <= -72.8)
        & (df["pickup_latitude"] >= 40.5)
        & (df["pickup_latitude"] <= 41.8)
        & (df["dropoff_longitude"] >= -74.5)
        & (df["dropoff_longitude"] <= -72.8)
        & (df["dropoff_latitude"] >= 40.5)
        & (df["dropoff_latitude"] <= 41.8)
        & (df["passenger_count"] >= 1)
        & (df["passenger_count"] <= 6)
    ).fillna(False)

    clean_train = df[mask].reset_index(drop=True)

    # Target and features
    y = clean_train["fare_amount"].skb.mark_as_y()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    X = clean_train.drop(columns=["fare_amount"]).skb.mark_as_X(
        cv=cv, split_kwargs={}
    )

    # Population check via adversarial validation on shared label-free observables
    test_df = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)
    adv_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    # Adversarial check on raw sample
    raw_sample = df[adv_cols].head(10000).assign(is_test=0)
    adv_test = test_df[adv_cols].assign(is_test=1)
    raw_combined = raw_sample.skb.concat([adv_test], axis=0).reset_index(drop=True)
    raw_X = raw_combined[adv_cols]
    raw_y = raw_combined["is_test"]
    raw_scores = skrub.as_data_op(
        HistGradientBoostingClassifier(random_state=42, max_iter=50)
    ).skb.apply_func(cross_val_score, raw_X, raw_y, cv=3, scoring="roc_auc")
    raw_auc = raw_scores.skb.apply_func(np.mean)

    # Adversarial check on clean sample
    clean_sample = clean_train[adv_cols].head(10000).assign(is_test=0)
    clean_combined = clean_sample.skb.concat([adv_test], axis=0).reset_index(drop=True)
    clean_X = clean_combined[adv_cols]
    clean_y = clean_combined["is_test"]
    clean_scores = skrub.as_data_op(
        HistGradientBoostingClassifier(random_state=42, max_iter=50)
    ).skb.apply_func(cross_val_score, clean_X, clean_y, cv=3, scoring="roc_auc")
    clean_auc = clean_scores.skb.apply_func(np.mean)

    audit = {
        "raw_rows_sampled": df.skb.apply_func(len),
        "clean_rows": clean_train.skb.apply_func(len),
        "adversarial_auc_raw": raw_auc,
        "adversarial_auc_clean": clean_auc,
    }

    return {
        "X": X,
        "y": y,
        "scoring": "neg_root_mean_squared_error",
        "audit": audit,
    }