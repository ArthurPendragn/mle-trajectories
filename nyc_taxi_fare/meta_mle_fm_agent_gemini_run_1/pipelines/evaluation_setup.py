import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

TRAIN_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
TEST_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/test.csv"
SAMPLE_SUBMISSION_PATH = (
    "gs://mle-trajectories-data/nyc_taxi_fare/input/sample_submission.csv"
)

# 500,000 rows provides a statistically robust sample (50x the test set size)
# while allowing fast 5-fold cross-validation iterations within run budgets.
N_ROWS = 500_000

FEATURE_COLS = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]


class AdversarialAUC(TransformerMixin, BaseEstimator):
    def __init__(self, max_iter=50):
        self.max_iter = max_iter

    def fit(self, X, y=None):
        clf = HistGradientBoostingClassifier(
            max_iter=self.max_iter, random_state=42
        )
        scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
        self.auc_mean_ = float(np.mean(scores))
        self.auc_std_ = float(np.std(scores))
        return self

    def transform(self, X):
        return pd.DataFrame(
            {
                "auc_mean": [self.auc_mean_] * len(X),
                "auc_std": [self.auc_std_] * len(X),
            }
        )


def build():
    train_raw = skrub.as_data_op(TRAIN_PATH).skb.apply_func(
        pd.read_csv, nrows=N_ROWS
    )
    test_raw = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    # Filter out corrupted entries and coordinates outside the NYC bounding box
    valid_mask = (
        (train_raw["fare_amount"] > 0.0)
        & (train_raw["fare_amount"] <= 500.0)
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

    clean_train = train_raw[valid_mask].reset_index(drop=True)

    # Adversarial validation comparing test set with candidate train populations
    test_features = test_raw[FEATURE_COLS].assign(is_test=1)
    raw_features = train_raw[FEATURE_COLS].iloc[:9914].assign(is_test=0)
    clean_features = clean_train[FEATURE_COLS].iloc[:9914].assign(is_test=0)

    adv_raw_table = test_features.skb.concat(
        [raw_features], axis=0
    ).reset_index(drop=True)
    adv_clean_table = test_features.skb.concat(
        [clean_features], axis=0
    ).reset_index(drop=True)

    adv_raw_auc = adv_raw_table[FEATURE_COLS].skb.apply(
        AdversarialAUC(max_iter=50), y=adv_raw_table["is_test"]
    )
    adv_clean_auc = adv_clean_table[FEATURE_COLS].skb.apply(
        AdversarialAUC(max_iter=50), y=adv_clean_table["is_test"]
    )

    raw_count = train_raw[["key"]].count()
    clean_count = clean_train[["key"]].count()

    audit = {
        "raw_count": raw_count,
        "clean_count": clean_count,
        "raw_adversarial_auc": adv_raw_auc,
        "clean_adversarial_auc": adv_clean_auc,
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    X = clean_train.drop(columns=["fare_amount"]).skb.mark_as_X(
        cv=cv, split_kwargs={}
    )
    y = clean_train["fare_amount"].skb.mark_as_y()

    return {
        "X": X,
        "y": y,
        "scoring": "neg_root_mean_squared_error",
        "audit": audit,
    }