import warnings
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score
import skrub

TRAIN_PATH = "/home/estrauss-ldap/repos/mle-trajectories/tab_playground_dec_21/input/train.csv"


class CVFoldSummaryEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=3, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            y_arr = X["Cover_Type"].values
        else:
            y_arr = np.asarray(X)

        n_samples = len(y_arr)
        all_classes = set(np.unique(y_arr))
        records = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_samples, dtype=np.int8), y_arr)):
                train_classes = set(np.unique(y_arr[train_idx]))
                test_classes = set(np.unique(y_arr[test_idx]))
                missing_train = sorted(list(all_classes - train_classes))
                missing_test = sorted(list(all_classes - test_classes))
                records.append({
                    "cv_method": "StratifiedKFold",
                    "fold": fold,
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                    "n_classes_train": len(train_classes),
                    "n_classes_test": len(test_classes),
                    "missing_classes_train": str(missing_train) if missing_train else "none",
                    "missing_classes_test": str(missing_test) if missing_test else "none",
                })

            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            for fold, (train_idx, test_idx) in enumerate(kf.split(np.zeros(n_samples, dtype=np.int8))):
                train_classes = set(np.unique(y_arr[train_idx]))
                test_classes = set(np.unique(y_arr[test_idx]))
                missing_train = sorted(list(all_classes - train_classes))
                missing_test = sorted(list(all_classes - test_classes))
                records.append({
                    "cv_method": "KFold",
                    "fold": fold,
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                    "n_classes_train": len(train_classes),
                    "n_classes_test": len(test_classes),
                    "missing_classes_train": str(missing_train) if missing_train else "none",
                    "missing_classes_test": str(missing_test) if missing_test else "none",
                })

        self.summary_df_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.summary_df_


class CVClassCountsEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_splits=3, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            y_arr = X["Cover_Type"].values
        else:
            y_arr = np.asarray(X)

        n_samples = len(y_arr)
        unique_classes = sorted(list(np.unique(y_arr)))
        records = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_samples, dtype=np.int8), y_arr)):
                train_s = pd.Series(y_arr[train_idx]).value_counts()
                test_s = pd.Series(y_arr[test_idx]).value_counts()
                for c in unique_classes:
                    tr_c = int(train_s.get(c, 0))
                    te_c = int(test_s.get(c, 0))
                    tot = tr_c + te_c
                    records.append({
                        "cv_method": "StratifiedKFold",
                        "fold": fold,
                        "cover_type": c,
                        "train_count": tr_c,
                        "test_count": te_c,
                        "train_pct": round(tr_c / (tot if tot > 0 else 1) * 100, 2),
                        "test_pct": round(te_c / (tot if tot > 0 else 1) * 100, 2),
                    })

            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            for fold, (train_idx, test_idx) in enumerate(kf.split(np.zeros(n_samples, dtype=np.int8))):
                train_s = pd.Series(y_arr[train_idx]).value_counts()
                test_s = pd.Series(y_arr[test_idx]).value_counts()
                for c in unique_classes:
                    tr_c = int(train_s.get(c, 0))
                    te_c = int(test_s.get(c, 0))
                    tot = tr_c + te_c
                    records.append({
                        "cv_method": "KFold",
                        "fold": fold,
                        "cover_type": c,
                        "train_count": tr_c,
                        "test_count": te_c,
                        "train_pct": round(tr_c / (tot if tot > 0 else 1) * 100, 2),
                        "test_pct": round(te_c / (tot if tot > 0 else 1) * 100, 2),
                    })

        self.counts_df_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.counts_df_


class SingletonLGBMEvalEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        feature_cols = [c for c in X.columns if c not in ("Id", "Cover_Type", "Soil_Type7", "Soil_Type15")]
        y_arr = X["Cover_Type"].values

        idx_class_5 = np.where(y_arr == 5)[0]
        idx_not_5 = np.where(y_arr != 5)[0]
        y_not_5 = y_arr[idx_not_5]

        # Draw a 60,000 sample subset stratified across classes 1, 2, 3, 4, 6, 7
        sub_idx, _ = train_test_split(
            idx_not_5,
            train_size=60000,
            stratify=y_not_5,
            random_state=self.random_state,
        )
        sub_y = y_arr[sub_idx]

        sub_tr_idx, sub_te_idx = train_test_split(
            sub_idx,
            train_size=0.5,
            stratify=sub_y,
            random_state=self.random_state,
        )

        records = []

        # Scenario A: Class 5 is in test set only (absent from train set)
        tr_idx_A = sub_tr_idx
        te_idx_A = np.concatenate([sub_te_idx, idx_class_5])

        X_tr_A = X.iloc[tr_idx_A][feature_cols]
        y_tr_A = y_arr[tr_idx_A]
        X_te_A = X.iloc[te_idx_A][feature_cols]
        y_te_A = y_arr[te_idx_A]

        clf_A = lgb.LGBMClassifier(
            n_estimators=40,
            num_leaves=31,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        clf_A.fit(X_tr_A, y_tr_A)
        preds_A = clf_A.predict(X_te_A)
        acc_A = accuracy_score(y_te_A, preds_A)
        proba_A = clf_A.predict_proba(X_te_A)

        records.append({
            "scenario": "class_5_in_test_only",
            "train_rows": len(tr_idx_A),
            "test_rows": len(te_idx_A),
            "train_has_class_5": bool(5 in y_tr_A),
            "test_has_class_5": bool(5 in y_te_A),
            "model_classes_count": len(clf_A.classes_),
            "predict_proba_cols": proba_A.shape[1],
            "fit_predict_status": "SUCCESS",
            "test_accuracy": round(float(acc_A), 5),
            "notes": "Accuracy evaluates cleanly; predict_proba outputs 6 class probabilities",
        })

        # Scenario B: Class 5 is in train set only (absent from test set)
        tr_idx_B = np.concatenate([sub_tr_idx, idx_class_5])
        te_idx_B = sub_te_idx

        X_tr_B = X.iloc[tr_idx_B][feature_cols]
        y_tr_B = y_arr[tr_idx_B]
        X_te_B = X.iloc[te_idx_B][feature_cols]
        y_te_B = y_arr[te_idx_B]

        clf_B = lgb.LGBMClassifier(
            n_estimators=40,
            num_leaves=31,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        clf_B.fit(X_tr_B, y_tr_B)
        preds_B = clf_B.predict(X_te_B)
        acc_B = accuracy_score(y_te_B, preds_B)
        proba_B = clf_B.predict_proba(X_te_B)

        records.append({
            "scenario": "class_5_in_train_only",
            "train_rows": len(tr_idx_B),
            "test_rows": len(te_idx_B),
            "train_has_class_5": bool(5 in y_tr_B),
            "test_has_class_5": bool(5 in y_te_B),
            "model_classes_count": len(clf_B.classes_),
            "predict_proba_cols": proba_B.shape[1],
            "fit_predict_status": "SUCCESS",
            "test_accuracy": round(float(acc_B), 5),
            "notes": "LightGBM cleanly fits 1 singleton sample; predict_proba outputs 7 class probabilities",
        })

        self.eval_df_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.eval_df_


class LGBMScalingStudyEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, sample_sizes=None, random_state=42):
        if sample_sizes is None:
            self.sample_sizes = [20000, 50000, 100000, 200000]
        else:
            self.sample_sizes = sample_sizes
        self.random_state = random_state

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        feature_cols = [c for c in X.columns if c not in ("Id", "Cover_Type", "Soil_Type7", "Soil_Type15")]
        y_arr = X["Cover_Type"].values

        idx_not_5 = np.where(y_arr != 5)[0]
        y_not_5 = y_arr[idx_not_5]

        # Fix a held-out 50,000 validation set to keep validation scores strictly comparable
        val_indices, pool_indices = train_test_split(
            idx_not_5,
            train_size=50000,
            stratify=y_not_5,
            random_state=self.random_state,
        )
        pool_y = y_arr[pool_indices]

        X_val = X.iloc[val_indices][feature_cols]
        y_val = y_arr[val_indices]

        records = []
        for n_train in self.sample_sizes:
            sub_tr_indices, _ = train_test_split(
                pool_indices,
                train_size=n_train,
                stratify=pool_y,
                random_state=self.random_state,
            )
            X_tr = X.iloc[sub_tr_indices][feature_cols]
            y_tr = y_arr[sub_tr_indices]

            clf = lgb.LGBMClassifier(
                n_estimators=50,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )

            t0 = time.perf_counter()
            clf.fit(X_tr, y_tr)
            fit_time = time.perf_counter() - t0

            val_preds = clf.predict(X_val)
            acc = accuracy_score(y_val, val_preds)

            records.append({
                "train_samples": n_train,
                "val_samples": len(val_indices),
                "fit_time_seconds": round(fit_time, 2),
                "throughput_samples_per_sec": round(n_train / max(fit_time, 0.001), 1),
                "val_accuracy": round(float(acc), 5),
            })

        self.scaling_df_ = pd.DataFrame(records)
        return self

    def transform(self, X):
        return self.scaling_df_


def build():
    train_df = skrub.as_data_op(TRAIN_PATH).skb.apply_func(pd.read_csv)

    fold_summary = train_df.skb.apply(CVFoldSummaryEstimator(n_splits=3, random_state=42))
    class_counts = train_df.skb.apply(CVClassCountsEstimator(n_splits=3, random_state=42))
    singleton_eval = train_df.skb.apply(SingletonLGBMEvalEstimator(random_state=42))
    scaling_study = train_df.skb.apply(LGBMScalingStudyEstimator(random_state=42))

    return {
        "fold_summary": fold_summary,
        "class_counts_per_fold": class_counts,
        "singleton_lgbm_eval": singleton_eval,
        "scaling_study": scaling_study,
    }