import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"
TEST_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/test.csv"

NUM_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

WILDERNESS_COLS = [
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
]

SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]

FEATURE_COLS = NUM_COLS + WILDERNESS_COLS + SOIL_COLS


class StratifiedKFoldAssigner(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=3, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for StratifiedKFoldAssigner")
        y_arr = np.asarray(y).ravel()
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        folds = np.zeros(len(y_arr), dtype=np.int64)
        for fold, (_, val_idx) in enumerate(skf.split(X, y_arr)):
            folds[val_idx] = fold
        self.folds_ = folds
        self.y_ = y_arr
        return self

    def transform(self, X):
        return pd.DataFrame({"Cover_Type": self.y_, "fold": self.folds_}, index=X.index)


def build():
    train = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    # 25% deterministic subsample of train (1,000,000 rows matching test set size)
    subsample = train[train["Id"] % 4 == 0]

    # Continuous feature distribution summaries
    train_num_summary = train[NUM_COLS].describe().round(2).reset_index()
    subsample_num_summary = subsample[NUM_COLS].describe().round(2).reset_index()
    test_num_summary = test[NUM_COLS].describe().round(2).reset_index()

    # Wilderness area distributions (proportions)
    train_wilderness_summary = train[WILDERNESS_COLS].describe().round(4).reset_index()
    subsample_wilderness_summary = subsample[WILDERNESS_COLS].describe().round(4).reset_index()
    test_wilderness_summary = test[WILDERNESS_COLS].describe().round(4).reset_index()

    # Soil type distributions (transposed for readability across all 40 columns)
    train_soil_summary = train[SOIL_COLS].describe().round(5).transpose().reset_index()
    test_soil_summary = test[SOIL_COLS].describe().round(5).transpose().reset_index()

    # Target distributions: verify subsample fidelity to full train population
    train_target_distribution = (
        train[["Cover_Type"]].value_counts(normalize=True).mul(100).round(4).reset_index()
    )
    subsample_target_distribution = (
        subsample[["Cover_Type"]].value_counts(normalize=True).mul(100).round(4).reset_index()
    )

    # Duplicate row analysis across all 54 feature columns and with target
    train_feature_duplicates = (
        train[FEATURE_COLS]
        .duplicated()
        .rename("is_duplicate")
        .value_counts()
        .reset_index()
    )
    train_exact_duplicates = (
        train[FEATURE_COLS + ["Cover_Type"]]
        .duplicated()
        .rename("is_duplicate")
        .value_counts()
        .reset_index()
    )
    test_feature_duplicates = (
        test[FEATURE_COLS]
        .duplicated()
        .rename("is_duplicate")
        .value_counts()
        .reset_index()
    )

    # Stratified 3-Fold test on full raw train (evaluating singleton class 5 handling)
    raw_folds = train[["Id"]].skb.apply(
        StratifiedKFoldAssigner(n_splits=3, shuffle=True, random_state=42),
        y=train["Cover_Type"],
    )
    skf_raw_fold_classes = (
        raw_folds.value_counts()
        .reset_index()
        .sort_values(["Cover_Type", "fold"])
        .reset_index(drop=True)
    )
    skf_raw_fold_counts = (
        raw_folds[["fold"]]
        .value_counts()
        .reset_index()
        .sort_values("fold")
        .reset_index(drop=True)
    )

    # Stratified 3-Fold test on filtered train (with singleton class 5 removed: 3,999,999 rows)
    train_filtered = train[train["Cover_Type"] != 5]
    filtered_folds = train_filtered[["Id"]].skb.apply(
        StratifiedKFoldAssigner(n_splits=3, shuffle=True, random_state=42),
        y=train_filtered["Cover_Type"],
    )
    skf_filtered_fold_classes = (
        filtered_folds.value_counts()
        .reset_index()
        .sort_values(["Cover_Type", "fold"])
        .reset_index(drop=True)
    )
    skf_filtered_fold_counts = (
        filtered_folds[["fold"]]
        .value_counts()
        .reset_index()
        .sort_values("fold")
        .reset_index(drop=True)
    )

    return {
        "train_num_summary": train_num_summary,
        "subsample_num_summary": subsample_num_summary,
        "test_num_summary": test_num_summary,
        "train_wilderness_summary": train_wilderness_summary,
        "subsample_wilderness_summary": subsample_wilderness_summary,
        "test_wilderness_summary": test_wilderness_summary,
        "train_soil_summary": train_soil_summary,
        "test_soil_summary": test_soil_summary,
        "train_target_distribution": train_target_distribution,
        "subsample_target_distribution": subsample_target_distribution,
        "train_feature_duplicates": train_feature_duplicates,
        "train_exact_duplicates": train_exact_duplicates,
        "test_feature_duplicates": test_feature_duplicates,
        "skf_raw_fold_classes": skf_raw_fold_classes,
        "skf_raw_fold_counts": skf_raw_fold_counts,
        "skf_filtered_fold_classes": skf_filtered_fold_classes,
        "skf_filtered_fold_counts": skf_filtered_fold_counts,
    }