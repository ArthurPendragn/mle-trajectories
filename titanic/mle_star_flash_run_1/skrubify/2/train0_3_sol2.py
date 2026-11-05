import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier


class GetXY(TransformerMixin, BaseEstimator):
    """Expose the CatBoost evaluation split inside each outer training fold."""

    def __init__(self, test_size=0.2, random_state=42):
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
            stratify=y,
        )
        return dict(zip(("X", "X_val", "y", "y_val"), parts))

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


class ProbabilityAverageClassifier(ClassifierMixin, BaseEstimator):
    """Preserve the original probability arithmetic and >= 0.5 threshold."""

    def fit(self, X, y, cb_X=None, cb_y=None, eval_set=None):
        self.classes_ = np.unique(y)

        # Construction inside this ensemble avoids CatBoost's cloning issues.
        self.cb_model_ = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            cat_features=["Sex", "Embarked", "Pclass"],
            eval_metric="Accuracy",
            random_seed=42,
            verbose=0,
        )
        self.xgb_model_ = XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            enable_categorical=True,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
        )
        self.lgb_model_ = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        self.cb_model_.fit(cb_X, cb_y, eval_set=eval_set, verbose=False)
        self.xgb_model_.fit(X, y)
        self.lgb_model_.fit(X, y)
        return self

    def predict(self, X):
        cb_probs = self.cb_model_.predict_proba(X)[:, 1]
        xgb_probs = self.xgb_model_.predict_proba(X)[:, 1]
        lgb_probs = self.lgb_model_.predict_proba(X)[:, 1]
        ensemble_probs = (cb_probs + xgb_probs + lgb_probs) / 3.0
        return (ensemble_probs >= 0.5).astype(int)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- record the CSV read without reading anything on import.
    data = skrub.as_data_op("./input/train.csv").skb.apply_func(pd.read_csv)

    # 2. Prepare data -- mark the raw target and retain the original feature order.
    #    StratifiedShuffleSplit reproduces the original stratified hold-out.
    y = data["Survived"].skb.mark_as_y()
    X = data[
        ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    ].skb.mark_as_X(
        cv=StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing, followed by the original three-model ensemble.
    features = X.assign(Embarked=X["Embarked"].fillna("Missing"))
    features = features.astype(
        {"Sex": "category", "Embarked": "category", "Pclass": "category"}
    )

    # CatBoost's eval_set implicitly enables best-iteration model selection even
    # without explicit early-stopping patience. The original selected that
    # iteration on its scored hold-out, leaking validation information.
    # Keep evaluation/model selection, but expose an inner stratified 20% split
    # of this fold's training rows instead. XGBoost and LightGBM still fit on all
    # outer-training rows. This necessarily differs from the original leaky score.
    cb_parts = features.skb.apply(GetXY(), y=y, no_wrap=True)

    pred = features.skb.apply(
        ProbabilityAverageClassifier(),
        y=y,
        fit_kwargs={
            "cb_X": cb_parts["X"],
            "cb_y": cb_parts["y"],
            "eval_set": (cb_parts["X_val"], cb_parts["y_val"]),
        },
    )

    # 4. Score -- only the splitter on mark_as_X controls outer validation.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring="accuracy"
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            f"Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
