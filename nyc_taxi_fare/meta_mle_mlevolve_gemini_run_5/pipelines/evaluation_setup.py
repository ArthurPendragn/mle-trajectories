import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
import skrub

TRAIN_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/train.csv.gz"
TEST_PATH = "gs://mle-trajectories-data/nyc_taxi_fare/input/test.csv"
SEED = 42
N_SPLITS = 5
N_ROWS = 500_000


class PopulationAuditor(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature_cols = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
        X_mat = X[feature_cols].to_numpy()
        y_vec = X["is_test"].to_numpy()

        clf = HistGradientBoostingClassifier(
            max_iter=50,
            random_state=self.random_state,
        )
        scores = cross_val_score(clf, X_mat, y_vec, cv=3, scoring="roc_auc")
        mean_auc = float(scores.mean())
        std_auc = float(scores.std())
        return pd.DataFrame(
            [
                {
                    "adversarial_auc_mean": mean_auc,
                    "adversarial_auc_std": std_auc,
                    "indistinguishable": bool(mean_auc < 0.55),
                    "subsample_size": N_ROWS,
                    "rationale": (
                        "Subsampled 500,000 rows from 55.4M to allow fast 5-fold CV iteration within "
                        "execution budget. Filtered coordinates to NYC bounding box and valid passenger counts, "
                        "aligning training population with test distribution."
                    ),
                }
            ]
        )


def build():
    raw_train = skrub.as_data_op(TRAIN_PATH).skb.apply_func(
        pd.read_csv, nrows=N_ROWS
    )
    test = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    valid_mask = (
        (raw_train["fare_amount"] >= 2.5).fillna(False)
        & (raw_train["fare_amount"] <= 500.0).fillna(False)
        & (raw_train["passenger_count"] >= 1).fillna(False)
        & (raw_train["passenger_count"] <= 6).fillna(False)
        & (raw_train["pickup_longitude"] >= -74.5).fillna(False)
        & (raw_train["pickup_longitude"] <= -72.8).fillna(False)
        & (raw_train["dropoff_longitude"] >= -74.5).fillna(False)
        & (raw_train["dropoff_longitude"] <= -72.8).fillna(False)
        & (raw_train["pickup_latitude"] >= 40.5).fillna(False)
        & (raw_train["pickup_latitude"] <= 41.8).fillna(False)
        & (raw_train["dropoff_latitude"] >= 40.5).fillna(False)
        & (raw_train["dropoff_latitude"] <= 41.8).fillna(False)
    )
    clean_train = raw_train[valid_mask].reset_index(drop=True)

    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    train_sample = clean_train[feature_cols].head(10_000).assign(is_test=0)
    test_sample = test[feature_cols].head(10_000).assign(is_test=1)
    adv_data = train_sample.skb.concat([test_sample], axis=0).reset_index(drop=True)
    audit_op = adv_data.skb.apply(PopulationAuditor(random_state=SEED))

    X_raw = clean_train[
        [
            "key",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ]
    y_raw = clean_train["fare_amount"]

    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    split_kwargs = {}

    X = X_raw.skb.mark_as_X(cv=cv, split_kwargs=split_kwargs)
    y = y_raw.skb.mark_as_y()

    scoring = "neg_root_mean_squared_error"

    return {
        "X": X,
        "y": y,
        "scoring": scoring,
        "audit": {"population_adversarial_check": audit_op},
    }