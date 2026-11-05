# Bound native worker pools before importing numerical libraries. The small
# Titanic folds otherwise incur substantial many-core scheduling overhead.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
import xgboost as xgb

from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits


class TicketFrequency(TransformerMixin, BaseEstimator):
    """Learn ticket frequencies from this fold's training rows."""

    def fit(self, X, y=None):
        self.counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        return X.assign(TicketGroupSize=X["Ticket"].map(self.counts_))


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Impute one column using training-only group medians."""

    def __init__(self, column, groups):
        self.column = column
        self.groups = groups

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.groups)[self.column].median()
        return self

    def transform(self, X):
        keys = X.set_index(self.groups).index
        fill_values = pd.Series(
            self.medians_.reindex(keys).to_numpy(),
            index=X.index,
        )
        return X.assign(
            **{self.column: X[self.column].fillna(fill_values)}
        )


class GetXY(TransformerMixin, BaseEstimator):
    """Expose an early-stopping split of the outer training rows."""

    def __init__(self, test_size=0.1, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        parts = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class OptimizedOOFEnsemble(ClassifierMixin, BaseEstimator):
    """Fit the original OOF-calibrated, fold-averaged three-model ensemble."""

    def __init__(self, n_splits=10, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y, eval_set):
        with threadpool_limits(limits=1):
            return self._fit_ensemble(X, y, eval_set)

    def _fit_ensemble(self, X, y, eval_set):
        y_array = np.asarray(y)
        self.classes_ = np.unique(y_array)
        self.n_features_in_ = X.shape[1]
        X_eval, y_eval = eval_set

        # These are INTERNAL calibration folds, not the outer scoring folds.
        # Indexing here is necessary to fit the OOF-calibrated ensemble without
        # learning its weights or threshold from outer validation labels.
        inner_cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        oof_lgb = np.zeros(len(X))
        oof_xgb = np.zeros(len(X))
        oof_cat = np.zeros(len(X))
        self.models_ = []

        for fold, (train_idx, val_idx) in enumerate(
            inner_cv.split(X, y_array)
        ):
            X_tr = X.iloc[train_idx]
            y_tr = y_array[train_idx]
            X_va = X.iloc[val_idx]

            # Only execution parallelism is explicitly limited. Model sizes,
            # learning parameters, patience, seeds, and fold counts are intact.
            lgb_model = lgb.LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state + fold,
                verbose=-1,
                n_jobs=1,
            )
            lgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_eval, y_eval)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False)
                ],
            )

            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                early_stopping_rounds=50,
                random_state=self.random_state + fold,
                n_jobs=1,
            )
            xgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_eval, y_eval)],
                verbose=False,
            )

            cat_model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.03,
                depth=5,
                subsample=0.8,
                eval_metric="Logloss",
                early_stopping_rounds=50,
                random_seed=self.random_state + fold,
                verbose=0,
                thread_count=1,
                allow_writing_files=False,
            )
            cat_model.fit(
                X_tr,
                y_tr,
                eval_set=(X_eval, y_eval),
                early_stopping_rounds=50,
                verbose=False,
            )

            oof_lgb[val_idx] = lgb_model.predict_proba(X_va)[:, 1]
            oof_xgb[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
            oof_cat[val_idx] = cat_model.predict_proba(X_va)[:, 1]
            self.models_.append((lgb_model, xgb_model, cat_model))

        # These are model outputs, not feature-matrix construction.
        oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat])

        def loss_func(params):
            w1, w2, w3, threshold = params
            weights = np.maximum(0, np.array([w1, w2, w3]))
            if weights.sum() == 0:
                return 0.0
            weights /= weights.sum()
            blend_probs = np.dot(oof_matrix, weights)
            predictions = (blend_probs >= threshold).astype(int)
            return -accuracy_score(y_array, predictions)

        result = minimize(
            loss_func,
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5],
            method="Nelder-Mead",
            bounds=[(0, 1), (0, 1), (0, 1), (0.3, 0.7)],
        )
        self.weights_ = np.maximum(0, result.x[:3])
        self.weights_ /= self.weights_.sum()
        self.threshold_ = result.x[3]
        return self

    def predict(self, X):
        with threadpool_limits(limits=1):
            lgb_probs = np.zeros(len(X))
            xgb_probs = np.zeros(len(X))
            cat_probs = np.zeros(len(X))

            # Preserve the original fold averaging and threshold decoding.
            for lgb_model, xgb_model, cat_model in self.models_:
                lgb_probs += lgb_model.predict_proba(X)[:, 1] / self.n_splits
                xgb_probs += xgb_model.predict_proba(X)[:, 1] / self.n_splits
                cat_probs += (
                    cat_model.predict_proba(X, thread_count=1)[:, 1]
                    / self.n_splits
                )

            probabilities = np.column_stack([lgb_probs, xgb_probs, cat_probs])
            blend = np.dot(probabilities, self.weights_)
            return (blend >= self.threshold_).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read.
    # Test prediction, submission writing, and the test-file existence check
    # are omitted. Preprocessing no longer learns from test.csv.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target without the original numpy cast.
    y = data["Survived"].skb.mark_as_y()
    X = data.drop(columns=["Survived"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        split_kwargs={},
    )

    # 3. Recorded feature engineering, preserving feature insertion order.
    title = X["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = title.replace(
        [
            "Lady", "Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona",
        ],
        "Rare",
    )
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    X = X.assign(Title=title.fillna("Missing"))

    X = X.assign(FamilySize=X["SibSp"] + X["Parch"] + 1)
    X = X.assign(IsAlone=(X["FamilySize"] == 1).astype(int))
    X = X.assign(
        SmallFamily=(
            (X["FamilySize"] >= 2) & (X["FamilySize"] <= 4)
        ).astype(int)
    )
    X = X.assign(LargeFamily=(X["FamilySize"] > 4).astype(int))
    X = X.assign(HasCabin=X["Cabin"].notna().astype(int))

    # pandas 3 may preserve nulls during string casts. Explicitly reproduce
    # the original missing-cabin stringification before extracting the deck.
    cabin_text = X["Cabin"].astype(str).fillna("nan")
    X = X.assign(
        CabinDeck=cabin_text.str[0].replace(
            {"n": "Missing", "T": "Missing"}
        )
    )

    # Independent learned statistics have separate estimator nodes. Fitting
    # per outer fold fixes the original global preprocessing leakage.
    X = X.skb.apply(TicketFrequency(), no_wrap=True)
    X = X.skb.apply(
        GroupMedianImputer(column="Fare", groups=["Pclass"]),
        no_wrap=True,
    )
    X = X.skb.apply(
        GroupMedianImputer(column="Age", groups=["Pclass", "Title"]),
        no_wrap=True,
    )
    X = X.skb.apply(
        SimpleImputer(strategy="median", keep_empty_features=True),
        cols=["Age"],
    )
    X = X.assign(Embarked=X["Embarked"].fillna("S"))
    X = X.assign(FarePerPerson=X["Fare"] / X["FamilySize"])
    X = X.assign(IsMinor=(X["Age"] < 16).astype(int))

    # Sorted string vocabularies match LabelEncoder on seen categories.
    # Explicit null stringification keeps this valid across pandas versions;
    # unseen categories receive -1 without consulting validation/test data.
    X = X.assign(
        Pclass=X["Pclass"].astype(str).fillna("nan"),
        Sex=X["Sex"].astype(str).fillna("nan"),
        Embarked=X["Embarked"].astype(str).fillna("nan"),
        Title=X["Title"].astype(str).fillna("nan"),
        CabinDeck=X["CabinDeck"].astype(str).fillna("nan"),
    )
    X = X.skb.apply(
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.int64,
        ),
        cols=["Pclass", "Sex", "Embarked", "Title", "CabinDeck"],
    )
    features = X.drop(
        columns=["PassengerId", "Name", "Ticket", "Cabin"],
        errors="ignore",
    )

    # The original early-stopped on scored rows, so its leaky score cannot be
    # reproduced honestly. Reserve 1/10 of outer-training rows, matching its
    # validation fraction. This shared eval split remains visible in the plan.
    pieces = features.skb.apply(
        GetXY(test_size=0.1, random_state=42),
        y=y,
        no_wrap=True,
    )
    X_fit = pieces["X"]
    y_fit = pieces.get("y", y)
    X_val = pieces["X_val"]
    y_val = pieces["y_val"]

    # The original also optimized weights/threshold on reported OOF labels.
    # Keep that optimization inside fitting using internal 10-fold OOF
    # predictions, and report accuracy on untouched outer validation rows.
    pred = X_fit.skb.apply(
        OptimizedOOFEnsemble(n_splits=10, random_state=42),
        y=y_fit,
        fit_kwargs={"eval_set": (X_val, y_val)},
    )

    # 4. Score -- mean outer-fold accuracy, not optimistically calibrated
    # pooled OOF accuracy. No cv= here: the marked splitter drives scoring.
    if __name__ == "__main__":
        with threadpool_limits(limits=1):
            search = pred.skb.make_grid_search(
                n_jobs=1,
                fitted=True,
                refit=False,
                scoring="accuracy",
            )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
