import time
import warnings
import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
import skrub

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"
TEST_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/test.csv"

CONTINUOUS_COLS = [
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
WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]
FEATURE_COLS = CONTINUOUS_COLS + WILDERNESS_COLS + SOIL_COLS

FEATURES_TO_TRACK = [
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Fire_Points",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
]


class ValidationSplitAuditor(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=3, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        records = []
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                skf = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                splits = list(skf.split(X, y))
                warn_msg = str(caught_warnings[-1].message) if caught_warnings else "None"
                class_5_in_all_train_folds = True
                for train_idx, _ in splits:
                    y_train = y.iloc[train_idx]
                    if not (y_train == 5).any():
                        class_5_in_all_train_folds = False
                        break
                records.append({
                    "splitter": "StratifiedKFold",
                    "status": "SUCCESS",
                    "n_splits": len(splits),
                    "warning_message": warn_msg,
                    "error_message": "None",
                    "class_5_in_all_train_folds": class_5_in_all_train_folds,
                })
            except Exception as exc:
                records.append({
                    "splitter": "StratifiedKFold",
                    "status": "FAILED",
                    "n_splits": 0,
                    "warning_message": "None",
                    "error_message": str(exc),
                    "class_5_in_all_train_folds": False,
                })

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                kf = KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                splits = list(kf.split(X, y))
                warn_msg = str(caught_warnings[-1].message) if caught_warnings else "None"
                class_5_in_all_train_folds = True
                for train_idx, _ in splits:
                    y_train = y.iloc[train_idx]
                    if not (y_train == 5).any():
                        class_5_in_all_train_folds = False
                        break
                records.append({
                    "splitter": "KFold",
                    "status": "SUCCESS",
                    "n_splits": len(splits),
                    "warning_message": warn_msg,
                    "error_message": "None",
                    "class_5_in_all_train_folds": class_5_in_all_train_folds,
                })
            except Exception as exc:
                records.append({
                    "splitter": "KFold",
                    "status": "FAILED",
                    "n_splits": 0,
                    "warning_message": "None",
                    "error_message": str(exc),
                    "class_5_in_all_train_folds": False,
                })

        self.results_df_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.results_df_


class LightGBMBenchmarkAuditor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sample_sizes=(100000, 500000, 1000000),
        n_estimators=20,
        random_state=42,
    ):
        self.sample_sizes = sample_sizes
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y=None):
        records = []
        for n_rows in self.sample_sizes:
            X_sub = X.iloc[:n_rows]
            y_sub = y.iloc[:n_rows]

            mem_mb = (
                X_sub.memory_usage(deep=True).sum() + y_sub.memory_usage(deep=True)
            ) / (1024 * 1024)

            clf = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
            t0 = time.perf_counter()
            clf.fit(X_sub, y_sub)
            elapsed_s = time.perf_counter() - t0

            throughput = n_rows / elapsed_s if elapsed_s > 0 else 0.0
            sec_per_iteration = elapsed_s / self.n_estimators

            records.append({
                "sample_size": n_rows,
                "n_estimators": self.n_estimators,
                "data_memory_mb": round(float(mem_mb), 2),
                "fit_duration_s": round(float(elapsed_s), 3),
                "sec_per_iteration": round(float(sec_per_iteration), 4),
                "throughput_rows_per_s": round(float(throughput), 1),
            })

        self.benchmark_df_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.benchmark_df_


def build():
    train_op = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)
    test_op = skrub.as_data_op(TEST_PATH).skb.apply_func(pd.read_csv)

    # 1. Feature drift across 1M-row train quartiles vs test
    train_block = train_op["Id"].floordiv(1000000).map({
        0: "train_0M_1M",
        1: "train_1M_2M",
        2: "train_2M_3M",
        3: "train_3M_4M",
    })
    train_means = (
        train_op.assign(block=train_block)
        .groupby("block")[FEATURES_TO_TRACK]
        .mean()
    )

    test_block = test_op["Id"].floordiv(1000000).map({
        4: "test_4M_5M",
    })
    test_means = (
        test_op.assign(block=test_block)
        .groupby("block")[FEATURES_TO_TRACK]
        .mean()
    )
    id_block_feature_drift = train_means.skb.concat([test_means], axis=0)

    # Target distribution across train quartiles
    id_block_target_drift = (
        train_op.assign(block=train_block)
        .groupby("block")["Cover_Type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        * 100
    )

    # 2. Duplicate rows and label consistency
    feat_dups = (
        train_op.duplicated(subset=FEATURE_COLS)
        .value_counts()
        .to_frame(name="feature_duplicates")
    )
    feat_label_dups = (
        train_op.duplicated(subset=FEATURE_COLS + ["Cover_Type"])
        .value_counts()
        .to_frame(name="feature_and_label_duplicates")
    )
    duplicate_summary = feat_dups.join(feat_label_dups)
    duplicate_analysis = duplicate_summary.assign(
        label_conflicts=duplicate_summary["feature_duplicates"]
        - duplicate_summary["feature_and_label_duplicates"]
    )

    # Singleton class 5 record inspection
    class_5_record = train_op[train_op["Cover_Type"] == 5][
        [
            "Id",
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Horizontal_Distance_To_Fire_Points",
            "Wilderness_Area1",
            "Wilderness_Area2",
            "Wilderness_Area3",
            "Wilderness_Area4",
            "Cover_Type",
        ]
    ]

    # 3. KFold vs StratifiedKFold validation check
    validation_audit = train_op[["Id"]].skb.apply(
        ValidationSplitAuditor(n_splits=3, random_state=42),
        y=train_op["Cover_Type"],
    )

    # 4. LightGBM throughput and memory benchmark across sample sizes
    benchmark_audit = train_op[FEATURE_COLS].skb.apply(
        LightGBMBenchmarkAuditor(
            sample_sizes=(100000, 500000, 1000000),
            n_estimators=20,
            random_state=42,
        ),
        y=train_op["Cover_Type"],
    )

    return {
        "id_block_feature_drift": id_block_feature_drift,
        "id_block_target_drift": id_block_target_drift,
        "duplicate_analysis": duplicate_analysis,
        "validation_splitter_audit": validation_audit,
        "lightgbm_throughput_benchmark": benchmark_audit,
        "singleton_class_5_record": class_5_record,
    }