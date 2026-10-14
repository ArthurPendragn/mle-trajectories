import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
import skrub

TRAIN_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
TEST_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/test.csv"
SAMPLE_ROWS = 500000


class AdversarialCheck(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y=None):
        features = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
        clf = HistGradientBoostingClassifier(
            max_iter=30, random_state=self.random_state
        )
        scores = cross_val_score(
            clf, X[features], X["is_test"], cv=3, scoring="roc_auc"
        )
        self.mean_auc_ = float(np.mean(scores))
        self.std_auc_ = float(np.std(scores))
        return self

    def transform(self, X):
        return pd.DataFrame(
            [
                {
                    "adversarial_roc_auc_mean": self.mean_auc_,
                    "adversarial_roc_auc_std": self.std_auc_,
                }
            ]
        )


def build():
    # Read the first 500,000 training rows deterministically from GCS.
    # The full training dataset has 55.4M rows (>10GB memory), which is too large
    # for rapid iteration and multiple CV experiments within budget.
    # 500k rows provides strong statistical power and low fold variance while
    # fitting models in under 30s.
    raw_train = skrub.as_data_op(TRAIN_PATH).skb.apply_func(
        pd.read_csv, nrows=SAMPLE_ROWS
    )
    test = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    # Filtering criteria based on valid NYC geographical bounding box and realistic fare amounts.
    valid_mask = (
        raw_train["fare_amount"].notna()
        & raw_train["pickup_longitude"].notna()
        & raw_train["pickup_latitude"].notna()
        & raw_train["dropoff_longitude"].notna()
        & raw_train["dropoff_latitude"].notna()
        & raw_train["passenger_count"].notna()
    )

    coord_fare_mask = (
        (raw_train["fare_amount"] >= 2.5)
        & (raw_train["fare_amount"] <= 500.0)
        & (raw_train["pickup_longitude"] >= -75.0)
        & (raw_train["pickup_longitude"] <= -72.5)
        & (raw_train["pickup_latitude"] >= 40.0)
        & (raw_train["pickup_latitude"] <= 42.0)
        & (raw_train["dropoff_longitude"] >= -75.0)
        & (raw_train["dropoff_longitude"] <= -72.5)
        & (raw_train["dropoff_latitude"] >= 40.0)
        & (raw_train["dropoff_latitude"] <= 42.0)
        & (raw_train["passenger_count"] >= 1)
        & (raw_train["passenger_count"] <= 6)
    )

    keep_mask = (valid_mask & coord_fare_mask).fillna(False)
    clean_train = raw_train[keep_mask].reset_index(drop=True)

    # Adversarial validation audit to verify population alignment between train and test
    test_features = test[
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ].assign(is_test=1)

    raw_features = (
        raw_train[
            [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
            ]
        ]
        .head(10000)
        .assign(is_test=0)
    )

    clean_features = (
        clean_train[
            [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
            ]
        ]
        .head(10000)
        .assign(is_test=0)
    )

    adv_raw = raw_features.skb.concat([test_features], axis=0).reset_index(drop=True)
    adv_clean = clean_features.skb.concat([test_features], axis=0).reset_index(drop=True)

    audit = {
        "count_raw": raw_train[["key"]].count(),
        "count_clean": clean_train[["key"]].count(),
        "adversarial_auc_raw": adv_raw.skb.apply(AdversarialCheck()),
        "adversarial_auc_clean": adv_clean.skb.apply(AdversarialCheck()),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    split_kwargs = {}

    X = clean_train.drop(columns=["fare_amount"]).skb.mark_as_X(
        cv=cv, split_kwargs=split_kwargs
    )
    y = clean_train["fare_amount"].skb.mark_as_y()
    scoring = "neg_root_mean_squared_error"

    return {
        "X": X,
        "y": y,
        "scoring": scoring,
        "audit": audit,
    }